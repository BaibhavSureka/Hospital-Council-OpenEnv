"""Open Gym-style environment for MIMIC-IV next-step clinical decision making.

This environment is intentionally designed for hackathon workflows:
- Loads a subset of MIMIC-IV hosp/icu schema tables.
- Defines 4 decision categories:
  0 diagnosis
  1 medication
  2 discharge
  3 no_action
- Supports dynamic tester queries and primary-agent actions.
- Executes every action inside the environment and validates it immediately.
- Triggers retrieval, memory, and web augmentation when an action diverges.
- Builds a structured LLM-ready state and converts the reasoning result to reward.
"""

from __future__ import annotations

import random
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple, Union
from urllib.parse import quote

import numpy as np
import pandas as pd
from pandas.errors import ParserError

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    try:
        import gym
        from gym import spaces
    except ImportError:  # pragma: no cover
        class _FallbackEnv:
            metadata: Dict[str, Any] = {}

        class _FallbackDiscrete:
            def __init__(self, n: int) -> None:
                self.n = int(n)

        class _FallbackBox:
            def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype: Any) -> None:
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class _FallbackSpaces:
            Box = _FallbackBox
            Discrete = _FallbackDiscrete

        class _FallbackGym:
            Env = _FallbackEnv

        gym = _FallbackGym()
        spaces = _FallbackSpaces()

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


CATEGORY_TO_ID: Dict[str, int] = {
    "diagnosis": 0,
    "medication": 1,
    "discharge": 2,
    "no_action": 3,
}

ID_TO_CATEGORY: Dict[int, str] = {v: k for k, v in CATEGORY_TO_ID.items()}

ActionInput = Union[int, np.integer, Dict[str, Any]]
ObserverFn = Callable[[Dict[str, Any]], Dict[str, Any]]
WebSearchFn = Callable[[str], Dict[str, Any]]
TesterAgentFn = Callable[[np.ndarray, Dict[str, Any]], Dict[str, Any]]
PrimaryAgentFn = Callable[[Dict[str, Any], np.ndarray, Dict[str, Any]], ActionInput]


@dataclass
class EncounterRecord:
    hadm_id: int
    subject_id: int
    los_hours: float
    in_icu: int
    expired_flag: int
    age: float
    diag_count: int
    med_count: int
    proc_count: int
    transfer_count: int
    meds: Tuple[str, ...]


@dataclass
class GroundTruthAction:
    category_id: int
    category: str
    action_text: str
    entities: Dict[str, Any]


@dataclass
class ReasoningObservation:
    classification: str
    probabilities: Dict[str, float]
    confidence: float
    observation: str


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\-\s]", " ", str(value).lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _token_set(value: str) -> Set[str]:
    return {token for token in _normalize_text(value).split(" ") if token}


def _jaccard_similarity(left: str, right: str) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return overlap / max(1, union)


class MIMICDecisionEnv(gym.Env):
    """Open Gym environment for category-level next-step clinical decisions."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_root: Union[str, Path],
        max_steps: int = 6,
        sample_size: int = 5000,
        table_row_limit: Optional[int] = None,
        trajectory_size: int = 25,
        llm_observer: Optional[ObserverFn] = None,
        reasoning_module: Optional[ObserverFn] = None,
        web_search_fn: Optional[WebSearchFn] = None,
        memory_size: int = 250,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.dataset_root = self._resolve_dataset_root(Path(data_root))
        self.max_steps = max(2, int(max_steps))
        self.sample_size = max(1, int(sample_size))
        self.table_row_limit = (
            max(10_000, self.sample_size * 50)
            if table_row_limit is None
            else max(1_000, int(table_row_limit))
        )

        self._rand = random.Random(seed)
        self._np_random = np.random.default_rng(seed)

        self.llm_observer = llm_observer
        self.reasoning_module = reasoning_module or llm_observer
        self.web_search_fn = web_search_fn or self._default_web_search

        self.trajectory: Deque[Dict[str, Any]] = deque(maxlen=max(5, int(trajectory_size)))
        self.interaction_memory: Deque[Dict[str, Any]] = deque(maxlen=max(25, int(memory_size)))
        self.records: List[EncounterRecord] = []
        self.medication_terms: List[str] = []
        self.medication_index: set[str] = set()
        self.hadm_to_meds: Dict[int, List[str]] = {}
        self.internal_action_db: Dict[str, Any] = {}
        self.category_aliases: Dict[str, List[str]] = {
            "diagnosis": ["diagnosis", "assess diagnosis", "evaluate condition", "workup"],
            "medication": ["medication", "give medication", "administer drug", "treat"],
            "discharge": ["discharge", "release patient", "send home", "transition care"],
            "no_action": ["no action", "continue monitoring", "observe", "watchful waiting"],
        }

        self.current_record: Optional[EncounterRecord] = None
        self.step_idx: int = 0
        self.last_reward: float = 0.0
        self.last_action_id: int = CATEGORY_TO_ID["no_action"]
        self.invalid_medication_last_step: int = 0
        self.last_category_probabilities: Dict[str, float] = self._uniform_category_probabilities()
        self.last_reasoning: ReasoningObservation = ReasoningObservation(
            classification="incorrect",
            probabilities=self._uniform_reasoning_probabilities(),
            confidence=0.0,
            observation="No reasoning has been run yet.",
        )

        self._load_data()

        # 12 state features + 4 probability features.
        self.observation_space = spaces.Box(
            low=-1_000_000.0,
            high=1_000_000.0,
            shape=(16,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

    def _resolve_dataset_root(self, data_root: Path) -> Path:
        if not data_root.exists():
            raise FileNotFoundError(f"Path not found: {data_root}")

        if (data_root / "hosp").exists() and (data_root / "icu").exists():
            return data_root

        for candidate in data_root.rglob("*"):
            if candidate.is_dir() and (candidate / "hosp").exists() and (candidate / "icu").exists():
                return candidate

        raise FileNotFoundError(
            "Could not find MIMIC folder containing both 'hosp/' and 'icu/'."
        )

    def _read_csv(
        self,
        rel_path: str,
        usecols: Optional[Sequence[str]] = None,
        parse_dates: Optional[Sequence[str]] = None,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        path = self.dataset_root / rel_path
        if not path.exists():
            columns = list(usecols) if usecols else []
            return pd.DataFrame(columns=columns)

        # Some mirrors use .csv.gz naming for plain-text CSV files.
        compression: Optional[str] = None
        try:
            with open(path, "rb") as fh:
                compression = "gzip" if fh.read(2) == b"\x1f\x8b" else None
        except OSError:
            compression = None

        kwargs = {
            "filepath_or_buffer": path,
            "usecols": list(usecols) if usecols else None,
            "parse_dates": list(parse_dates) if parse_dates else None,
            "compression": compression,
            "low_memory": False,
            "nrows": nrows,
        }

        try:
            return pd.read_csv(**kwargs)
        except (ParserError, MemoryError):
            # Fallback for malformed rows or tokenizer pressure on large files.
            return pd.read_csv(
                **kwargs,
                engine="python",
                on_bad_lines="skip",
            )

    def _load_data(self) -> None:
        admissions = self._read_csv(
            "hosp/admissions.csv.gz",
            usecols=[
                "subject_id",
                "hadm_id",
                "admittime",
                "dischtime",
                "hospital_expire_flag",
            ],
            parse_dates=["admittime", "dischtime"],
            nrows=max(self.sample_size * 3, 20_000),
        )
        if admissions.empty:
            raise ValueError("No admissions data found. Check your dataset path.")

        patients = self._read_csv(
            "hosp/patients.csv.gz",
            usecols=["subject_id", "anchor_age"],
            nrows=self.table_row_limit,
        )
        diagnoses = self._read_csv(
            "hosp/diagnoses_icd.csv.gz", usecols=["hadm_id"], nrows=self.table_row_limit
        )
        procedures = self._read_csv(
            "hosp/procedures_icd.csv.gz", usecols=["hadm_id"], nrows=self.table_row_limit
        )
        transfers = self._read_csv(
            "hosp/transfers.csv.gz", usecols=["hadm_id"], nrows=self.table_row_limit
        )
        icustays = self._read_csv(
            "icu/icustays.csv.gz", usecols=["hadm_id"], nrows=self.table_row_limit
        )

        prescriptions = self._read_csv(
            "hosp/prescriptions.csv.gz",
            usecols=["hadm_id", "drug"],
            nrows=self.table_row_limit,
        ).rename(columns={"drug": "medication_text"})
        pharmacy = self._read_csv(
            "hosp/pharmacy.csv.gz",
            usecols=["hadm_id", "medication"],
            nrows=self.table_row_limit,
        ).rename(columns={"medication": "medication_text"})
        emar = self._read_csv(
            "hosp/emar.csv.gz",
            usecols=["hadm_id", "medication"],
            nrows=self.table_row_limit,
        ).rename(columns={"medication": "medication_text"})

        admissions = admissions.dropna(subset=["hadm_id", "subject_id"])
        admissions["hadm_id"] = admissions["hadm_id"].astype(int)
        admissions["subject_id"] = admissions["subject_id"].astype(int)

        diag_counts = diagnoses.groupby("hadm_id").size() if not diagnoses.empty else pd.Series(dtype="int64")
        proc_counts = procedures.groupby("hadm_id").size() if not procedures.empty else pd.Series(dtype="int64")
        transfer_counts = (
            transfers.groupby("hadm_id").size() if not transfers.empty else pd.Series(dtype="int64")
        )

        med_frames = [df for df in (prescriptions, pharmacy, emar) if not df.empty]
        if med_frames:
            med_all = pd.concat(med_frames, ignore_index=True)
            med_all = med_all.dropna(subset=["hadm_id", "medication_text"])
            med_all["hadm_id"] = med_all["hadm_id"].astype(int)
            med_all["medication_text"] = med_all["medication_text"].astype(str)
            med_counts = med_all.groupby("hadm_id").size()
            meds_by_hadm = med_all.groupby("hadm_id")["medication_text"].apply(list).to_dict()
        else:
            med_counts = pd.Series(dtype="int64")
            meds_by_hadm = {}

        self.medication_terms = self._build_medication_terms(med_frames)
        self.medication_index = set(self.medication_terms)

        icu_hadm_ids = set(icustays["hadm_id"].dropna().astype(int).tolist()) if not icustays.empty else set()

        patient_age = {}
        if not patients.empty:
            patient_age = {
                _safe_int(row.subject_id): _safe_float(row.anchor_age)
                for row in patients.itertuples(index=False)
            }

        # Keep loading lightweight for fast iteration in hacking workflows.
        if len(admissions) > self.sample_size:
            admissions = admissions.sample(n=self.sample_size, random_state=42)

        records: List[EncounterRecord] = []
        for row in admissions.itertuples(index=False):
            hadm_id = _safe_int(row.hadm_id)
            subject_id = _safe_int(row.subject_id)

            los_hours = 24.0
            if pd.notna(row.admittime) and pd.notna(row.dischtime):
                dt = row.dischtime - row.admittime
                los_hours = max(dt.total_seconds() / 3600.0, 0.5)

            meds = tuple(str(m) for m in meds_by_hadm.get(hadm_id, [])[:5])
            records.append(
                EncounterRecord(
                    hadm_id=hadm_id,
                    subject_id=subject_id,
                    los_hours=los_hours,
                    in_icu=1 if hadm_id in icu_hadm_ids else 0,
                    expired_flag=_safe_int(getattr(row, "hospital_expire_flag", 0)),
                    age=patient_age.get(subject_id, 0.0),
                    diag_count=int(diag_counts.get(hadm_id, 0)),
                    med_count=int(med_counts.get(hadm_id, 0)),
                    proc_count=int(proc_counts.get(hadm_id, 0)),
                    transfer_count=int(transfer_counts.get(hadm_id, 0)),
                    meds=meds,
                )
            )

        if not records:
            raise ValueError("No encounter records could be built from MIMIC data.")

        self.records = records
        self.hadm_to_meds = {
            _safe_int(hadm_id): [str(med) for med in meds[:10]]
            for hadm_id, meds in meds_by_hadm.items()
        }
        self._build_internal_action_db()

    def _build_internal_action_db(self) -> None:
        medication_entities = {
            term: {
                "entity_type": "medication",
                "category": "medication",
                "aliases": [term.split(" ")[0]] if " " in term else [term],
            }
            for term in self.medication_terms
        }
        category_entities = {
            category: {
                "entity_type": "category",
                "category": category,
                "aliases": aliases,
            }
            for category, aliases in self.category_aliases.items()
        }
        encounter_entities = {
            record.hadm_id: {
                "subject_id": record.subject_id,
                "medications": [_normalize_text(med) for med in record.meds if _normalize_text(med)],
                "diag_count": record.diag_count,
                "med_count": record.med_count,
                "proc_count": record.proc_count,
                "in_icu": record.in_icu,
                "expired_flag": record.expired_flag,
            }
            for record in self.records
        }
        self.internal_action_db = {
            "categories": category_entities,
            "medications": medication_entities,
            "encounters": encounter_entities,
        }

    def _build_medication_terms(self, med_frames: Sequence[pd.DataFrame]) -> List[str]:
        terms: set[str] = set()
        for frame in med_frames:
            if frame.empty or "medication_text" not in frame.columns:
                continue
            for raw in frame["medication_text"].dropna().astype(str).tolist():
                norm = _normalize_text(raw)
                if len(norm) >= 3:
                    terms.add(norm)
                # Also index first token for flexible matching (vitamin -> vitamin c etc.).
                first = norm.split(" ")[0] if norm else ""
                if len(first) >= 3:
                    terms.add(first)
        return sorted(terms)

    def _uniform_category_probabilities(self) -> Dict[str, float]:
        return {k: 0.25 for k in CATEGORY_TO_ID}

    def _uniform_reasoning_probabilities(self) -> Dict[str, float]:
        return {
            "correct": 1.0 / 3.0,
            "partially_correct": 1.0 / 3.0,
            "incorrect": 1.0 / 3.0,
        }

    def _parse_action(self, action: ActionInput) -> Dict[str, Any]:
        if isinstance(action, (int, np.integer)):
            category_id = int(action)
            raw_action = ID_TO_CATEGORY.get(category_id, "no_action")
            tester_prompt = ""
        elif isinstance(action, dict):
            category_val = action.get("category", action.get("category_id", CATEGORY_TO_ID["no_action"]))
            if isinstance(category_val, str):
                category_id = CATEGORY_TO_ID.get(_normalize_text(category_val), CATEGORY_TO_ID["no_action"])
            else:
                category_id = _safe_int(category_val, CATEGORY_TO_ID["no_action"])

            raw_action = str(action.get("agent_response", action.get("action_text", ""))).strip()
            if not raw_action:
                raw_action = ID_TO_CATEGORY.get(category_id, "no_action")
            tester_prompt = str(action.get("tester_prompt", "")).strip()
        else:
            raise ValueError("Action must be int or dict payload.")

        if category_id not in ID_TO_CATEGORY:
            category_id = CATEGORY_TO_ID["no_action"]

        med_term = self._extract_medication_term(raw_action)

        return {
            "category_id": category_id,
            "category": ID_TO_CATEGORY[category_id],
            "raw_action": raw_action,
            "tester_prompt": tester_prompt,
            "medication_term": med_term,
        }

    def _normalize_tester_case(self, tester_output: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(tester_output, dict):
            tester_output = {"query": str(tester_output)}

        query = str(
            tester_output.get("query", tester_output.get("tester_prompt", info.get("tester_query", "")))
        ).strip()
        ground_truth = self._ground_truth_action(self.current_record, self.step_idx) if self.current_record else None
        return {
            "query": query or info.get("tester_query", ""),
            "expected_action": tester_output.get(
                "expected_action",
                ground_truth.action_text if ground_truth else info.get("ground_truth_action", ""),
            ),
            "ground_truth_category": tester_output.get(
                "ground_truth_category",
                ground_truth.category if ground_truth else info.get("ground_truth_category", "no_action"),
            ),
            "metadata": dict(tester_output.get("metadata", {})),
        }

    def _compose_action_payload(
        self,
        tester_case: Dict[str, Any],
        action: ActionInput,
    ) -> ActionInput:
        if not isinstance(action, dict):
            if isinstance(action, (int, np.integer)):
                category_id = int(action)
            else:
                category_id = CATEGORY_TO_ID["no_action"]
            return {
                "category_id": category_id,
                "tester_prompt": tester_case.get("query", ""),
                "agent_response": ID_TO_CATEGORY.get(category_id, "no_action"),
            }

        payload = dict(action)
        payload.setdefault("tester_prompt", tester_case.get("query", ""))
        raw_action = str(
            payload.get("agent_response", payload.get("action_text", payload.get("raw_action", "")))
        ).strip()
        if not raw_action:
            raw_action = tester_case.get("query", "")
            payload["agent_response"] = raw_action
        return payload

    def _extract_medication_term(self, action_text: str) -> str:
        text = _normalize_text(action_text)
        if not text:
            return ""

        match = re.search(r"(?:give|administer|start|use|provide)\s+([a-z0-9\-\s]+)", text)
        term = match.group(1).strip() if match else text

        # Keep up to 4 tokens to reduce noise.
        tokens = [tok for tok in term.split(" ") if tok]
        return " ".join(tokens[:4])

    def lookup_medication(self, term: str, limit: int = 5) -> Dict[str, Any]:
        term_norm = _normalize_text(term)
        if not term_norm:
            return {"term": term_norm, "found": False, "matches": []}

        if term_norm in self.medication_index:
            return {"term": term_norm, "found": True, "matches": [term_norm]}

        matches = [m for m in self.medication_terms if term_norm in m or m in term_norm][:limit]
        return {"term": term_norm, "found": len(matches) > 0, "matches": matches}

    def _ground_truth_action(
        self,
        record: Optional[EncounterRecord],
        step_idx: int,
    ) -> GroundTruthAction:
        if record is None:
            return GroundTruthAction(
                category_id=CATEGORY_TO_ID["no_action"],
                category="no_action",
                action_text="continue monitoring",
                entities={},
            )

        category_id = self._ground_truth_for_step(record, step_idx)
        category = ID_TO_CATEGORY[category_id]
        if category == "medication":
            medication = _normalize_text(record.meds[0]) if record.meds else "supportive medication"
            medication = medication or "supportive medication"
            return GroundTruthAction(
                category_id=category_id,
                category=category,
                action_text=f"give {medication}",
                entities={"medication": medication},
            )
        if category == "diagnosis":
            return GroundTruthAction(
                category_id=category_id,
                category=category,
                action_text="perform diagnostic assessment",
                entities={"focus": "diagnostic assessment"},
            )
        if category == "discharge":
            return GroundTruthAction(
                category_id=category_id,
                category=category,
                action_text="prepare discharge plan",
                entities={"focus": "discharge planning"},
            )
        return GroundTruthAction(
            category_id=category_id,
            category=category,
            action_text="continue monitoring",
            entities={"focus": "monitoring"},
        )

    def _state_snapshot(
        self,
        record: Optional[EncounterRecord] = None,
    ) -> Dict[str, Any]:
        record = record or self.current_record
        if record is None:
            return {
                "hadm_id": None,
                "subject_id": None,
                "los_hours": 0.0,
                "in_icu": 0,
                "expired_flag": 0,
                "age": 0.0,
                "diag_count": 0,
                "med_count": 0,
                "proc_count": 0,
                "transfer_count": 0,
                "candidate_medications": [],
            }

        return {
            "hadm_id": record.hadm_id,
            "subject_id": record.subject_id,
            "los_hours": round(record.los_hours, 3),
            "in_icu": record.in_icu,
            "expired_flag": record.expired_flag,
            "age": round(record.age, 1),
            "diag_count": record.diag_count,
            "med_count": record.med_count,
            "proc_count": record.proc_count,
            "transfer_count": record.transfer_count,
            "candidate_medications": [
                med for med in [_normalize_text(item) for item in record.meds] if med
            ][:5],
        }

    def build_primary_agent_context(
        self,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        info = info or {}
        step_index = _safe_int(info.get("step_index", self.step_idx))
        state = self._state_snapshot()
        return {
            "step_index": step_index,
            "step_progress": step_index / max(1, self.max_steps - 1),
            "max_steps": self.max_steps,
            "state": state,
            "recent_trajectory": self.fetch_trajectory(3),
            "memory_size": len(self.interaction_memory),
        }

    def _prepare_primary_agent_case(self, tester_case: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(tester_case.get("metadata", {}))
        metadata.pop("ground_truth_category", None)
        metadata.pop("expected_action", None)
        return {
            "query": str(tester_case.get("query", "")).strip(),
            "metadata": metadata,
        }

    def generate_tester_query(
        self,
        record: Optional[EncounterRecord] = None,
        step_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        record = record or self.current_record
        step_idx = self.step_idx if step_idx is None else int(step_idx)
        ground_truth = self._ground_truth_action(record, step_idx)
        category = ground_truth.category
        state = self._state_snapshot(record)
        if category == "medication":
            medication = ground_truth.entities.get("medication", "supportive medication")
            templates = [
                "something active may need to start here; what would you do next",
                "the case feels like it needs more than observation now",
                "what is the next concrete move if waiting is no longer enough",
                f"there are {state['med_count']} treatment signals already in play; what should happen next",
            ]
        elif category == "diagnosis":
            templates = [
                "the picture is still unclear; what should happen next",
                "before acting too aggressively, what step would you take first",
                "we still do not have enough clarity; what is the next move",
                f"the case has {state['diag_count']} documented diagnoses so far; what would you do now",
            ]
        elif category == "discharge":
            templates = [
                "things may be wrapping up; what is the next move",
                "does this look like a point to transition out of the stay",
                "if the acute phase is settling, what should happen next",
                f"the stay is at step {step_idx + 1} of {self.max_steps}; what action fits now",
            ]
        else:
            templates = [
                "would you change anything yet, or hold steady",
                "is this a moment for action or for watching",
                "nothing is clearly forcing a move; what would you do now",
                "does it make sense to stay conservative at this point",
            ]

        return {
            "query": self._rand.choice(templates),
            "expected_action": ground_truth.action_text,
            "ground_truth_category": category,
            "metadata": {
                "hadm_id": record.hadm_id if record else None,
                "subject_id": record.subject_id if record else None,
                "step_idx": step_idx,
                "candidate_medications": state["candidate_medications"][:2],
                "query_style": "very_vague",
            },
        }

    def default_primary_agent(
        self,
        tester_case: Dict[str, Any],
        _obs: np.ndarray,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        query = str(tester_case.get("query", "")).strip()
        query_norm = _normalize_text(query)
        context_state = dict(info.get("state", {}))
        step_progress = _safe_float(info.get("step_progress", 0.0))
        candidate_meds = [str(item) for item in context_state.get("candidate_medications", []) if str(item).strip()]

        if any(token in query_norm for token in ("transition", "wrapping", "settling", "out of the stay")):
            category = "discharge"
            response = "prepare discharge plan"
        elif (
            step_progress >= 0.80
            and not _safe_int(context_state.get("in_icu", 0))
            and not _safe_int(context_state.get("expired_flag", 0))
        ):
            category = "discharge"
            response = "prepare discharge plan"
        elif (
            _safe_int(context_state.get("med_count", 0)) > 0
            and (
                step_progress >= 0.34
                or any(token in query_norm for token in ("active", "concrete", "waiting", "treatment", "start"))
            )
        ):
            category = "medication"
            medication = candidate_meds[0] if candidate_meds else self._extract_medication_term(query)
            response = f"give {medication or 'supportive medication'}"
        elif any(
            token in query_norm
            for token in (
                "hold steady",
                "watching",
                "conservative",
                "nothing is screaming",
                "or hold steady",
                "for now",
                "change anything yet",
            )
        ):
            category = "no_action"
            response = "continue monitoring"
        elif (
            _safe_int(context_state.get("diag_count", 0)) == 0
            or (
                _safe_int(context_state.get("proc_count", 0)) > 0
                and step_progress < 0.80
            )
            or any(token in query_norm for token in ("unclear", "clarity", "first", "picture"))
        ):
            category = "diagnosis"
            response = "perform diagnostic assessment"
        else:
            category = "no_action"
            response = "continue monitoring"

        return {
            "category": category,
            "agent_response": response,
            "tester_prompt": query,
        }

    def validate_action_against_db(
        self,
        parsed_action: Dict[str, Any],
        ground_truth: GroundTruthAction,
    ) -> Dict[str, Any]:
        encounter_db = (
            self.internal_action_db.get("encounters", {}).get(self.current_record.hadm_id, {})
            if self.current_record
            else {}
        )
        related_entities = {
            "encounter_medications": encounter_db.get("medications", []),
            "category_aliases": self.category_aliases.get(parsed_action["category"], []),
        }

        if parsed_action["category"] == "medication":
            medication_lookup = self.lookup_medication(parsed_action["medication_term"])
            expected_medication = str(ground_truth.entities.get("medication", ""))
            related_entities["expected_medication"] = expected_medication
            related_entities["matching_encounter_medications"] = [
                med
                for med in encounter_db.get("medications", [])
                if medication_lookup.get("term", "") and medication_lookup["term"] in med
            ]
            return {
                "exists": bool(medication_lookup.get("found", False)),
                "lookup_term": medication_lookup.get("term", ""),
                "matches": medication_lookup.get("matches", []),
                "entity_type": "medication",
                "related_entities": related_entities,
                "mapping_found": expected_medication in medication_lookup.get("matches", []),
            }

        category_entry = self.internal_action_db.get("categories", {}).get(parsed_action["category"], {})
        return {
            "exists": bool(category_entry),
            "lookup_term": parsed_action["category"],
            "matches": category_entry.get("aliases", []),
            "entity_type": "category",
            "related_entities": related_entities,
            "mapping_found": parsed_action["category"] == ground_truth.category,
        }

    def _default_web_search(self, query: str) -> Dict[str, Any]:
        if not query:
            return {"status": "skipped", "reason": "empty query"}
        if requests is None:
            return {"status": "skipped", "reason": "requests package is not installed"}

        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(query)}"
        try:
            res = requests.get(url, timeout=3)
            if res.status_code != 200:
                return {
                    "status": "error",
                    "code": int(res.status_code),
                    "query": query,
                    "summary": "",
                }
            payload = res.json()
            return {
                "status": "ok",
                "query": query,
                "title": payload.get("title", ""),
                "description": payload.get("description", ""),
                "summary": str(payload.get("extract", ""))[:500],
                "source": payload.get("content_urls", {}).get("desktop", {}).get("page", ""),
            }
        except Exception as exc:  # pragma: no cover
            return {"status": "error", "query": query, "error": str(exc), "summary": ""}

    def retrieve_similar_trajectories(
        self,
        query: str,
        parsed_action: Dict[str, Any],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in self.interaction_memory:
            item_query = str(item.get("query", ""))
            item_action = item.get("agent_action", {})
            query_score = _jaccard_similarity(query, item_query)
            action_score = _jaccard_similarity(
                parsed_action.get("raw_action", ""),
                str(item_action.get("raw_action", "")),
            )
            category_bonus = 0.15 if parsed_action.get("category") == item_action.get("category") else 0.0
            score = query_score * 0.55 + action_score * 0.30 + category_bonus
            if score <= 0.0:
                continue
            scored.append(
                (
                    score,
                    {
                        "similarity": round(score, 4),
                        "query": item_query,
                        "action": item_action,
                        "outcome": item.get("outcome", {}),
                        "reward": item.get("reward", 0.0),
                        "time_step": item.get("time_step", -1),
                    },
                )
            )

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[: max(1, int(limit))]]

    def build_web_context(self, query: str, parsed_action: Dict[str, Any]) -> Dict[str, Any]:
        candidates: List[str] = []
        for candidate in (query, parsed_action.get("raw_action", ""), parsed_action.get("medication_term", "")):
            candidate = str(candidate).strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        if not candidates:
            return {"status": "skipped", "reason": "no search terms", "results": []}

        results = [self.web_search_fn(candidate) for candidate in candidates[:3]]
        return {"status": "ok", "results": results}

    def fetch_trajectory(self, k: int = 5) -> List[Dict[str, Any]]:
        k = max(1, int(k))
        return list(self.trajectory)[-k:]

    def _ground_truth_for_step(self, record: EncounterRecord, step_idx: int) -> int:
        progress = step_idx / max(1, self.max_steps - 1)

        if progress < 0.34:
            if record.diag_count == 0:
                return CATEGORY_TO_ID["diagnosis"]
            if record.med_count > 0:
                return CATEGORY_TO_ID["medication"]
            return CATEGORY_TO_ID["no_action"]

        if progress < 0.80:
            if record.med_count > 0:
                return CATEGORY_TO_ID["medication"]
            if record.proc_count > 0:
                return CATEGORY_TO_ID["diagnosis"]
            return CATEGORY_TO_ID["no_action"]

        if record.expired_flag == 0 and record.in_icu == 0:
            return CATEGORY_TO_ID["discharge"]
        if record.med_count > 0:
            return CATEGORY_TO_ID["medication"]
        return CATEGORY_TO_ID["no_action"]

    def _action_matches_ground_truth(
        self,
        parsed_action: Dict[str, Any],
        ground_truth: GroundTruthAction,
    ) -> bool:
        if parsed_action["category"] != ground_truth.category:
            return False
        if ground_truth.category != "medication":
            return True

        expected_medication = _normalize_text(str(ground_truth.entities.get("medication", "")))
        if not expected_medication:
            return True

        actual_medication = _normalize_text(parsed_action.get("medication_term", ""))
        if actual_medication == expected_medication:
            return True

        lookup = self.lookup_medication(actual_medication)
        return expected_medication in lookup.get("matches", [])

    def _estimate_category_probabilities(
        self,
        ground_truth: GroundTruthAction,
        parsed_action: Dict[str, Any],
        database_validation: Dict[str, Any],
        exact_match: bool,
    ) -> Dict[str, float]:
        probs = self._uniform_category_probabilities()
        probs[ground_truth.category] += 0.35
        probs[parsed_action["category"]] += 0.20
        if exact_match:
            probs[ground_truth.category] += 0.30
        if parsed_action["category"] == "medication" and database_validation.get("exists", False):
            probs["medication"] += 0.15
        if self.step_idx / max(1, self.max_steps - 1) > 0.80:
            probs["discharge"] += 0.10

        total = sum(probs.values()) or 1.0
        return {key: float(value / total) for key, value in probs.items()}

    def _build_structured_state(
        self,
        query: str,
        parsed_action: Dict[str, Any],
        ground_truth: GroundTruthAction,
        database_validation: Dict[str, Any],
        retrieved_trajectories: List[Dict[str, Any]],
        web_context: Dict[str, Any],
        exact_match: bool,
        medication_lookup: Dict[str, Any],
        elapsed_ms: float,
    ) -> Dict[str, Any]:
        current_state = {
            "hadm_id": self.current_record.hadm_id if self.current_record else None,
            "subject_id": self.current_record.subject_id if self.current_record else None,
            "los_hours": round(self.current_record.los_hours, 3) if self.current_record else 0.0,
            "in_icu": self.current_record.in_icu if self.current_record else 0,
            "expired_flag": self.current_record.expired_flag if self.current_record else 0,
            "diag_count": self.current_record.diag_count if self.current_record else 0,
            "med_count": self.current_record.med_count if self.current_record else 0,
            "proc_count": self.current_record.proc_count if self.current_record else 0,
            "transfer_count": self.current_record.transfer_count if self.current_record else 0,
        }
        return {
            "query": query,
            "agent_action": parsed_action,
            "ground_truth": {
                "category_id": ground_truth.category_id,
                "category": ground_truth.category,
                "action_text": ground_truth.action_text,
                "entities": ground_truth.entities,
            },
            "database_validation": database_validation,
            "retrieved_trajectories": retrieved_trajectories,
            "web_context": web_context,
            "medication_lookup": medication_lookup,
            "time_step": self.step_idx,
            "step_progress": self.step_idx / max(1, self.max_steps - 1),
            "execution": {
                "exact_match": exact_match,
                "category_match": parsed_action["category"] == ground_truth.category,
                "action_time_ms": round(elapsed_ms, 3),
            },
            "state": current_state,
        }

    def _heuristic_reasoning_module(self, structured_state: Dict[str, Any]) -> ReasoningObservation:
        probs = self._uniform_reasoning_probabilities()
        query = str(structured_state.get("query", ""))
        agent_action = structured_state.get("agent_action", {})
        ground_truth = structured_state.get("ground_truth", {})
        database_validation = structured_state.get("database_validation", {})
        retrieved_trajectories = structured_state.get("retrieved_trajectories", [])
        execution = structured_state.get("execution", {})
        web_results = structured_state.get("web_context", {}).get("results", [])

        exact_match = bool(execution.get("exact_match", False))
        category_match = bool(execution.get("category_match", False))
        action_text = str(agent_action.get("raw_action", ""))
        ground_truth_text = str(ground_truth.get("action_text", ""))
        semantic_similarity = max(
            _jaccard_similarity(query, action_text),
            _jaccard_similarity(ground_truth_text, action_text),
        )

        trajectory_support = 0.0
        if retrieved_trajectories:
            support_scores = []
            for item in retrieved_trajectories:
                outcome = item.get("outcome", {})
                category = item.get("action", {}).get("category")
                classification = outcome.get("classification", "incorrect")
                score = 0.0
                if category == agent_action.get("category"):
                    if classification == "correct":
                        score = 1.0
                    elif classification == "partially_correct":
                        score = 0.6
                    else:
                        score = 0.1
                support_scores.append(score)
            trajectory_support = float(sum(support_scores) / max(1, len(support_scores)))

        web_support = any(
            result.get("status") == "ok" and result.get("summary")
            for result in web_results
            if isinstance(result, dict)
        )

        if exact_match:
            probs["correct"] += 0.70
        elif category_match:
            probs["partially_correct"] += 0.45
            probs["incorrect"] += 0.10
        else:
            probs["incorrect"] += 0.45

        if semantic_similarity >= 0.80:
            probs["correct"] += 0.10
        elif semantic_similarity >= 0.45:
            probs["partially_correct"] += 0.20
        else:
            probs["incorrect"] += 0.10

        if database_validation.get("exists", False):
            probs["partially_correct"] += 0.12
            if database_validation.get("mapping_found", False):
                probs["correct"] += 0.08
        else:
            probs["incorrect"] += 0.12

        if trajectory_support >= 0.60:
            probs["correct"] += 0.10
        elif retrieved_trajectories:
            probs["incorrect"] += 0.08

        if web_support:
            probs["partially_correct"] += 0.05

        total = sum(probs.values()) or 1.0
        normalized = {key: float(value / total) for key, value in probs.items()}
        classification = max(normalized, key=normalized.get)
        confidence = float(normalized[classification])
        observation = (
            f"Action classified as {classification} "
            f"(exact_match={exact_match}, semantic_similarity={semantic_similarity:.3f}, "
            f"trajectory_support={trajectory_support:.3f})."
        )
        return ReasoningObservation(
            classification=classification,
            probabilities=normalized,
            confidence=confidence,
            observation=observation,
        )

    def _call_reasoning_module(self, structured_state: Dict[str, Any]) -> ReasoningObservation:
        if self.reasoning_module is None:
            return self._heuristic_reasoning_module(structured_state)

        try:
            out = self.reasoning_module(structured_state)
            out_dict = out if isinstance(out, dict) else {}
            probs = {}
            if out_dict:
                probs = out_dict.get("class_probabilities", out_dict.get("probabilities", {}))

            if set(probs).issubset(CATEGORY_TO_ID.keys()) and probs:
                top_category = max(probs, key=probs.get)
                ground_truth_category = structured_state.get("ground_truth", {}).get("category", "no_action")
                if top_category == ground_truth_category:
                    probs = {"correct": 0.72, "partially_correct": 0.18, "incorrect": 0.10}
                elif structured_state.get("agent_action", {}).get("category") == ground_truth_category:
                    probs = {"correct": 0.22, "partially_correct": 0.58, "incorrect": 0.20}
                else:
                    probs = {"correct": 0.08, "partially_correct": 0.22, "incorrect": 0.70}

            for label in ("correct", "partially_correct", "incorrect"):
                probs.setdefault(label, 0.0)

            total = sum(float(value) for value in probs.values()) or 1.0
            normalized = {key: float(value) / total for key, value in probs.items()}
            classification = str(out_dict.get("classification", max(normalized, key=normalized.get)))
            if classification not in normalized:
                classification = max(normalized, key=normalized.get)
            confidence = float(out_dict.get("confidence", normalized[classification]))
            return ReasoningObservation(
                classification=classification,
                probabilities=normalized,
                confidence=confidence,
                observation=str(out_dict.get("observation", out_dict.get("rationale", ""))),
            )
        except Exception as exc:  # pragma: no cover
            fallback = self._heuristic_reasoning_module(structured_state)
            fallback.observation = f"{fallback.observation} Observer error: {exc}"
            return fallback

    def _compute_reward(
        self,
        reasoning: ReasoningObservation,
        exact_match: bool,
    ) -> float:
        reward_map = {
            "correct": 1.00,
            "partially_correct": 0.25,
            "incorrect": -0.75,
        }
        reward = reward_map.get(reasoning.classification, -0.50)
        reward += 0.20 * reasoning.confidence
        if exact_match:
            reward += 0.10
        reward -= 0.01
        return float(max(-1.0, min(1.5, reward)))

    def _build_observation(self) -> np.ndarray:
        if self.current_record is None:
            return np.zeros((16,), dtype=np.float32)

        probs = self.last_category_probabilities
        rec = self.current_record
        obs = np.array(
            [
                rec.los_hours / 24.0,
                float(rec.in_icu),
                float(rec.expired_flag),
                rec.diag_count / 20.0,
                rec.med_count / 20.0,
                rec.proc_count / 20.0,
                rec.transfer_count / 20.0,
                self.step_idx / max(1, self.max_steps),
                self.last_reward,
                float(self.invalid_medication_last_step),
                self.last_action_id / 3.0,
                rec.age / 100.0,
                probs["diagnosis"],
                probs["medication"],
                probs["discharge"],
                probs["no_action"],
            ],
            dtype=np.float32,
        )
        return obs

    def _build_reset_info(self) -> Dict[str, Any]:
        if self.current_record is None:
            return {}
        ground_truth = self._ground_truth_action(self.current_record, self.step_idx)
        tester_case = self.generate_tester_query(self.current_record, self.step_idx)
        return {
            "hadm_id": self.current_record.hadm_id,
            "subject_id": self.current_record.subject_id,
            "ground_truth_id": ground_truth.category_id,
            "ground_truth_category": ground_truth.category,
            "ground_truth_action": ground_truth.action_text,
            "tester_query": tester_case["query"],
            "trajectory": self.fetch_trajectory(),
            "agent_context": self.build_primary_agent_context(),
            "memory_size": len(self.interaction_memory),
            "categories": CATEGORY_TO_ID.copy(),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rand.seed(seed)
            self._np_random = np.random.default_rng(seed)

        self.current_record = self._rand.choice(self.records)
        self.step_idx = 0
        self.last_reward = 0.0
        self.last_action_id = CATEGORY_TO_ID["no_action"]
        self.invalid_medication_last_step = 0
        self.last_category_probabilities = self._uniform_category_probabilities()
        self.last_reasoning = ReasoningObservation(
            classification="incorrect",
            probabilities=self._uniform_reasoning_probabilities(),
            confidence=0.0,
            observation="No reasoning has been run yet.",
        )
        self.trajectory.clear()

        obs = self._build_observation()
        info = self._build_reset_info()
        return obs, info

    def step(
        self, action: ActionInput
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_record is None:
            raise RuntimeError("Call reset() before step().")

        t0 = time.perf_counter()
        parsed = self._parse_action(action)
        ground_truth = self._ground_truth_action(self.current_record, self.step_idx)
        tester_query = parsed["tester_prompt"] or self.generate_tester_query(
            self.current_record,
            self.step_idx,
        )["query"]
        exact_match = self._action_matches_ground_truth(parsed, ground_truth)
        med_lookup = {"term": "", "found": False, "matches": []}
        if parsed["category"] == "medication":
            med_lookup = self.lookup_medication(parsed["medication_term"])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if exact_match:
            database_validation = {
                "status": "skipped",
                "reason": "exact ground-truth match",
                "exists": True,
                "mapping_found": True,
            }
            retrieved_trajectories = []
            web_context = {"status": "skipped", "reason": "exact ground-truth match", "results": []}
        else:
            database_validation = self.validate_action_against_db(parsed, ground_truth)
            retrieved_trajectories = self.retrieve_similar_trajectories(tester_query, parsed)
            web_context = self.build_web_context(tester_query, parsed)

        structured_state = self._build_structured_state(
            query=tester_query,
            parsed_action=parsed,
            ground_truth=ground_truth,
            database_validation=database_validation,
            retrieved_trajectories=retrieved_trajectories,
            web_context=web_context,
            exact_match=exact_match,
            medication_lookup=med_lookup,
            elapsed_ms=elapsed_ms,
        )
        reasoning = self._call_reasoning_module(structured_state)
        reward = self._compute_reward(reasoning, exact_match)
        category_probabilities = self._estimate_category_probabilities(
            ground_truth=ground_truth,
            parsed_action=parsed,
            database_validation=database_validation,
            exact_match=exact_match,
        )

        trajectory_entry = {
            "step": self.step_idx,
            "query": tester_query,
            "ground_truth": ground_truth.category,
            "ground_truth_action": ground_truth.action_text,
            "action": parsed["category"],
            "raw_action": parsed["raw_action"],
            "reward": round(reward, 4),
            "classification": reasoning.classification,
            "time_taken_ms": round(elapsed_ms, 3),
            "medication_lookup": med_lookup,
        }
        self.trajectory.append(trajectory_entry)

        memory_entry = {
            "time_step": self.step_idx,
            "query": tester_query,
            "agent_action": parsed,
            "ground_truth": {
                "category": ground_truth.category,
                "action_text": ground_truth.action_text,
                "entities": ground_truth.entities,
            },
            "state": structured_state,
            "reward": reward,
            "outcome": {
                "classification": reasoning.classification,
                "confidence": reasoning.confidence,
                "observation": reasoning.observation,
                "exact_match": exact_match,
            },
        }
        self.interaction_memory.append(memory_entry)

        self.last_reward = reward
        self.last_action_id = parsed["category_id"]
        self.invalid_medication_last_step = int(
            parsed["category"] == "medication" and not med_lookup.get("found", False)
        )
        self.last_category_probabilities = category_probabilities
        self.last_reasoning = reasoning

        self.step_idx += 1

        terminated = self.step_idx >= self.max_steps
        if parsed["category"] == "discharge":
            terminated = True
        truncated = False

        next_obs = self._build_observation()
        agent_context = self.build_primary_agent_context({"step_index": self.step_idx})
        info = {
            "hadm_id": self.current_record.hadm_id,
            "subject_id": self.current_record.subject_id,
            "step_index": self.step_idx,
            "tester_query": tester_query,
            "ground_truth_id": ground_truth.category_id,
            "ground_truth_category": ground_truth.category,
            "ground_truth_action": ground_truth.action_text,
            "decision_id": parsed["category_id"],
            "decision_category": parsed["category"],
            "decision_action": parsed["raw_action"],
            "exact_match": exact_match,
            "medication_lookup": med_lookup,
            "database_validation": database_validation,
            "retrieved_trajectories": retrieved_trajectories,
            "web_search": web_context,
            "trajectory": self.fetch_trajectory(),
            "agent_context": agent_context,
            "structured_state": structured_state,
            "llm_payload": structured_state,
            "llm_observation": reasoning.observation,
            "llm_probabilities": reasoning.probabilities,
            "reasoning_classification": reasoning.classification,
            "reasoning_confidence": reasoning.confidence,
            "category_probabilities": category_probabilities,
            "action_time_ms": round(elapsed_ms, 3),
            "reward": reward,
            "memory_size": len(self.interaction_memory),
            "true_label": ground_truth.category_id,
        }
        return next_obs, reward, terminated, truncated, info

    def run_agent_episode(
        self,
        tester_agent: Optional[TesterAgentFn] = None,
        primary_agent: Optional[PrimaryAgentFn] = None,
        *,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        obs, info = self.reset(seed=seed)
        tester_agent = tester_agent or (lambda obs_val, info_val: self.generate_tester_query(self.current_record, self.step_idx))
        primary_agent = primary_agent or self.default_primary_agent
        terminated = False
        truncated = False
        interactions: List[Dict[str, Any]] = []

        while not (terminated or truncated):
            tester_case = self._normalize_tester_case(tester_agent(obs, info), info)
            primary_case = self._prepare_primary_agent_case(tester_case)
            primary_context = self.build_primary_agent_context(info)
            action = primary_agent(primary_case, obs, primary_context)
            payload = self._compose_action_payload(tester_case, action)
            obs, reward, terminated, truncated, info = self.step(payload)
            step_record = dict(info)
            step_record["episode_reward_so_far"] = round(
                sum(float(item.get("reward", 0.0)) for item in interactions) + float(reward),
                4,
            )
            interactions.append(step_record)

        return interactions

    def run_continuous_loop(
        self,
        tester_agent: Optional[TesterAgentFn] = None,
        primary_agent: Optional[PrimaryAgentFn] = None,
        episodes: int = 3,
        *,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        episode_logs: List[List[Dict[str, Any]]] = []
        total_reward = 0.0
        for offset in range(max(1, int(episodes))):
            episode = self.run_agent_episode(
                tester_agent=tester_agent,
                primary_agent=primary_agent,
                seed=None if seed is None else seed + offset,
            )
            episode_logs.append(episode)
            total_reward += sum(float(step.get("reward", 0.0)) for step in episode)

        return {
            "episodes": episode_logs,
            "avg_reward": total_reward / max(1, len(episode_logs)),
            "memory_size": len(self.interaction_memory),
        }

    def render(self) -> None:
        if self.current_record is None:
            print("Environment not reset.")
            return
        print(
            {
                "hadm_id": self.current_record.hadm_id,
                "step": self.step_idx,
                "last_reward": round(self.last_reward, 4),
                "last_action": ID_TO_CATEGORY.get(self.last_action_id, "unknown"),
                "last_reasoning": self.last_reasoning.classification,
            }
        )


def _classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int],
) -> Dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred lengths do not match.")

    confusion = {
        true_label: {pred_label: 0 for pred_label in labels}
        for true_label in labels
    }

    correct = 0
    for truth, pred in zip(y_true, y_pred):
        if truth in confusion and pred in confusion[truth]:
            confusion[truth][pred] += 1
        if truth == pred:
            correct += 1

    total = max(1, len(y_true))
    metrics = {
        "accuracy": correct / total,
        "support": len(y_true),
        "per_category": {},
        "confusion": confusion,
    }

    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[t][label] for t in labels if t != label)
        fn = sum(confusion[label][p] for p in labels if p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        support = sum(confusion[label].values())
        metrics["per_category"][ID_TO_CATEGORY[label]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return metrics


def evaluate_next_step_predictions(
    env: MIMICDecisionEnv,
    policy_fn: Callable[[np.ndarray, Dict[str, Any]], ActionInput],
    episodes: int = 10,
) -> Dict[str, Any]:
    y_true: List[int] = []
    y_pred: List[int] = []
    rewards: List[float] = []
    exact_matches = 0
    reasoning_correct = 0
    medication_support = 0

    for _ in range(max(1, int(episodes))):
        obs, info = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0.0

        while not (terminated or truncated):
            action = policy_fn(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            y_true.append(_safe_int(info.get("ground_truth_id", CATEGORY_TO_ID["no_action"])))
            y_pred.append(_safe_int(info.get("decision_id", CATEGORY_TO_ID["no_action"])))
            exact_matches += int(bool(info.get("exact_match", False)))
            reasoning_correct += int(str(info.get("reasoning_classification", "")) == "correct")
            medication_support += int(
                _safe_int(info.get("ground_truth_id", CATEGORY_TO_ID["no_action"]))
                == CATEGORY_TO_ID["medication"]
            )
            ep_reward += float(reward)

        rewards.append(ep_reward)

    metrics = _classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=[
            CATEGORY_TO_ID["diagnosis"],
            CATEGORY_TO_ID["medication"],
            CATEGORY_TO_ID["discharge"],
            CATEGORY_TO_ID["no_action"],
        ],
    )
    support = max(1, len(y_true))
    metrics["category_accuracy"] = metrics.get("accuracy", 0.0)
    metrics["exact_match_rate"] = exact_matches / support
    metrics["reasoning_correct_rate"] = reasoning_correct / support
    metrics["medication_support"] = medication_support
    metrics["avg_reward"] = float(sum(rewards) / max(1, len(rewards)))
    metrics["episodes"] = int(episodes)
    return metrics


def format_metrics(metrics: Dict[str, Any]) -> str:
    lines = []
    lines.append("=== Next Step Prediction Metrics ===")
    lines.append(f"Episodes: {metrics.get('episodes', 0)}")
    lines.append(f"Support: {metrics.get('support', 0)}")
    lines.append(f"Category Accuracy: {metrics.get('category_accuracy', metrics.get('accuracy', 0.0)):.4f}")
    lines.append(f"Exact Match Rate: {metrics.get('exact_match_rate', 0.0):.4f}")
    lines.append(f"Reasoning-Correct Rate: {metrics.get('reasoning_correct_rate', 0.0):.4f}")
    lines.append(f"Medication Case Support: {metrics.get('medication_support', 0)}")
    lines.append(f"Average Reward: {metrics.get('avg_reward', 0.0):.4f}")
    lines.append("")
    lines.append("Per Category:")
    for name, vals in metrics.get("per_category", {}).items():
        lines.append(
            (
                f"- {name}: precision={vals['precision']:.4f}, "
                f"recall={vals['recall']:.4f}, f1={vals['f1']:.4f}, support={vals['support']}"
            )
        )
    return "\n".join(lines)
