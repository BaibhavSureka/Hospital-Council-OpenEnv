"""Open Gym-style environment for MIMIC-IV next-step clinical decision making.

This environment is intentionally designed for hackathon workflows:
- Loads a subset of MIMIC-IV hosp/icu schema tables.
- Defines 4 decision categories:
  0 diagnosis
  1 medication
  2 discharge
  3 no_action
- Supports dynamic action payloads from an external tester/agent.
- Validates medication mentions against MIMIC medication fields.
- Returns trajectory + web-search context + LLM-ready payload in info.
"""

from __future__ import annotations

import random
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import quote

import numpy as np
import pandas as pd
from pandas.errors import ParserError

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces

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
        web_search_fn: Optional[WebSearchFn] = None,
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
        self.web_search_fn = web_search_fn or self._default_web_search

        self.trajectory: Deque[Dict[str, Any]] = deque(maxlen=max(5, int(trajectory_size)))
        self.records: List[EncounterRecord] = []
        self.medication_terms: List[str] = []
        self.medication_index: set[str] = set()

        self.current_record: Optional[EncounterRecord] = None
        self.step_idx: int = 0
        self.last_reward: float = 0.0
        self.last_action_id: int = CATEGORY_TO_ID["no_action"]
        self.invalid_medication_last_step: int = 0
        self.last_llm_probabilities: Dict[str, float] = self._uniform_probabilities()

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

    def _uniform_probabilities(self) -> Dict[str, float]:
        return {k: 0.25 for k in CATEGORY_TO_ID}

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

    def _compute_reward(
        self,
        action_id: int,
        ground_truth_id: int,
        med_lookup: Dict[str, Any],
        tester_prompt: str,
    ) -> float:
        reward = 1.0 if action_id == ground_truth_id else -0.5

        if ID_TO_CATEGORY[action_id] == "medication":
            reward += 0.4 if med_lookup.get("found", False) else -0.4

        if tester_prompt:
            expected = self._extract_medication_term(tester_prompt)
            actual = med_lookup.get("term", "")
            if expected and actual and expected != actual:
                reward -= 0.2

        # Small step cost to incentivize shorter trajectories.
        reward -= 0.01
        return float(reward)

    def _heuristic_llm_observer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        probs = self._uniform_probabilities()

        ground_truth = payload.get("ground_truth", "no_action")
        current_action = payload.get("current_action", {}).get("category", "no_action")
        med_found = bool(payload.get("medication_lookup", {}).get("found", False))
        progress = _safe_float(payload.get("step_progress", 0.0))

        if ground_truth in probs:
            probs[ground_truth] += 0.30
        if current_action in probs:
            probs[current_action] += 0.10
        if med_found:
            probs["medication"] += 0.15
        if progress > 0.80:
            probs["discharge"] += 0.10

        total = sum(probs.values()) or 1.0
        probs = {k: float(v / total) for k, v in probs.items()}

        top = max(probs, key=probs.get)
        observation = (
            f"Likely next category is '{top}' with probability {probs[top]:.3f}. "
            f"Ground truth reference for this step is '{ground_truth}'."
        )
        return {"observation": observation, "probabilities": probs}

    def _call_llm_observer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm_observer is None:
            return self._heuristic_llm_observer(payload)

        try:
            out = self.llm_observer(payload)
            probs = out.get("probabilities", {}) if isinstance(out, dict) else {}
            for name in CATEGORY_TO_ID:
                probs.setdefault(name, 0.0)
            total = sum(float(v) for v in probs.values()) or 1.0
            probs = {k: float(v) / total for k, v in probs.items()}
            return {
                "observation": str(out.get("observation", "")),
                "probabilities": probs,
            }
        except Exception as exc:  # pragma: no cover
            fallback = self._heuristic_llm_observer(payload)
            fallback["observer_error"] = str(exc)
            return fallback

    def _build_observation(self) -> np.ndarray:
        if self.current_record is None:
            return np.zeros((16,), dtype=np.float32)

        probs = self.last_llm_probabilities
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
        gt = self._ground_truth_for_step(self.current_record, self.step_idx)
        return {
            "hadm_id": self.current_record.hadm_id,
            "subject_id": self.current_record.subject_id,
            "ground_truth_id": gt,
            "ground_truth_category": ID_TO_CATEGORY[gt],
            "trajectory": self.fetch_trajectory(),
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
        self.last_llm_probabilities = self._uniform_probabilities()
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

        gt_id = self._ground_truth_for_step(self.current_record, self.step_idx)
        med_lookup = {"term": "", "found": False, "matches": []}
        web_info = {"status": "skipped", "reason": "not medication action"}

        if parsed["category"] == "medication":
            med_lookup = self.lookup_medication(parsed["medication_term"])
            if med_lookup.get("term"):
                web_info = self.web_search_fn(med_lookup["term"])

        reward = self._compute_reward(
            action_id=parsed["category_id"],
            ground_truth_id=gt_id,
            med_lookup=med_lookup,
            tester_prompt=parsed["tester_prompt"],
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        trajectory_entry = {
            "step": self.step_idx,
            "ground_truth": ID_TO_CATEGORY[gt_id],
            "action": parsed["category"],
            "raw_action": parsed["raw_action"],
            "reward": round(reward, 4),
            "time_taken_ms": round(elapsed_ms, 3),
            "medication_lookup": med_lookup,
        }
        self.trajectory.append(trajectory_entry)

        llm_payload = {
            "ground_truth": ID_TO_CATEGORY[gt_id],
            "current_action": {
                "category": parsed["category"],
                "raw_action": parsed["raw_action"],
                "tester_prompt": parsed["tester_prompt"],
            },
            "time_taken_ms": round(elapsed_ms, 3),
            "fetch_trajectory": self.fetch_trajectory(),
            "web_search": web_info,
            "medication_lookup": med_lookup,
            "step_progress": self.step_idx / max(1, self.max_steps - 1),
            "state": {
                "hadm_id": self.current_record.hadm_id,
                "subject_id": self.current_record.subject_id,
                "los_hours": round(self.current_record.los_hours, 3),
                "in_icu": self.current_record.in_icu,
                "expired_flag": self.current_record.expired_flag,
                "diag_count": self.current_record.diag_count,
                "med_count": self.current_record.med_count,
                "proc_count": self.current_record.proc_count,
                "transfer_count": self.current_record.transfer_count,
            },
        }

        llm_out = self._call_llm_observer(llm_payload)

        self.last_reward = reward
        self.last_action_id = parsed["category_id"]
        self.invalid_medication_last_step = int(
            parsed["category"] == "medication" and not med_lookup.get("found", False)
        )
        self.last_llm_probabilities = llm_out.get("probabilities", self._uniform_probabilities())

        self.step_idx += 1

        terminated = self.step_idx >= self.max_steps
        if parsed["category"] == "discharge":
            terminated = True
        truncated = False

        next_obs = self._build_observation()
        info = {
            "hadm_id": self.current_record.hadm_id,
            "subject_id": self.current_record.subject_id,
            "step_index": self.step_idx,
            "ground_truth_id": gt_id,
            "ground_truth_category": ID_TO_CATEGORY[gt_id],
            "decision_id": parsed["category_id"],
            "decision_category": parsed["category"],
            "medication_lookup": med_lookup,
            "web_search": web_info,
            "trajectory": self.fetch_trajectory(),
            "llm_payload": llm_payload,
            "llm_observation": llm_out.get("observation", ""),
            "llm_probabilities": llm_out.get("probabilities", {}),
            "action_time_ms": round(elapsed_ms, 3),
            "reward": reward,
            "true_label": gt_id,
        }
        return next_obs, reward, terminated, truncated, info

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
    metrics["avg_reward"] = float(sum(rewards) / max(1, len(rewards)))
    metrics["episodes"] = int(episodes)
    return metrics


def format_metrics(metrics: Dict[str, Any]) -> str:
    lines = []
    lines.append("=== Next Step Prediction Metrics ===")
    lines.append(f"Episodes: {metrics.get('episodes', 0)}")
    lines.append(f"Support: {metrics.get('support', 0)}")
    lines.append(f"Accuracy: {metrics.get('accuracy', 0.0):.4f}")
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
