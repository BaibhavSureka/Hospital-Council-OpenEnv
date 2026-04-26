# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core simulator for the Hospital Council OpenEnv environment."""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from pandas.errors import ParserError

try:
    from .augmentation import ContextLLMManager, LLMSearchAugmenter, timestamp_ms
except ImportError:
    from augmentation import ContextLLMManager, LLMSearchAugmenter, timestamp_ms


CATEGORY_TO_ID: Dict[str, int] = {
    "diagnosis": 0,
    "medication": 1,
    "discharge": 2,
    "no_action": 3,
}

SCENARIO_TYPES: Tuple[str, ...] = (
    "diagnostic_ambiguity",
    "medication_alignment",
    "conservative_monitoring",
    "discharge_negotiation",
)

STAKEHOLDER_DESCRIPTIONS: Dict[str, str] = {
    "attending_physician": "Owns clinical direction and resists premature closure.",
    "triage_nurse": "Tracks bedside uncertainty, symptom drift, and practical feasibility.",
    "pharmacist": "Guards medication safety, interaction risk, and treatment specificity.",
    "bed_manager": "Optimizes bed flow, step-down timing, and throughput pressure.",
    "family_liaison": "Focuses on communication quality, expectation management, and trust.",
}

PRIMARY_TARGETS: Dict[str, str] = {
    "diagnostic_ambiguity": "attending_physician",
    "medication_alignment": "pharmacist",
    "conservative_monitoring": "triage_nurse",
    "discharge_negotiation": "bed_manager",
}

SECONDARY_TARGETS: Dict[str, str] = {
    "diagnostic_ambiguity": "triage_nurse",
    "medication_alignment": "attending_physician",
    "conservative_monitoring": "family_liaison",
    "discharge_negotiation": "family_liaison",
}


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
    lab_event_count: int
    abnormal_lab_event_count: int
    salient_labs: Tuple[str, ...]
    salient_lab_categories: Tuple[str, ...]


@dataclass
class CouncilStage:
    step_index: int
    phase_name: str
    expected_action_type: str
    expected_category: str
    preferred_targets: Tuple[str, ...]
    rationale: str


@dataclass
class StakeholderState:
    name: str
    alignment: float
    influence: float
    consulted: int = 0
    last_note: str = ""


@dataclass
class EpisodeSnapshot:
    scenario_id: str
    scenario_type: str
    difficulty: str
    record: EncounterRecord
    mission_brief: str
    long_horizon_goals: List[str]
    visible_conflicts: List[str]
    patient_snapshot: Dict[str, Any]
    stages: List[CouncilStage]
    stakeholders: Dict[str, StakeholderState]
    step_count: int = 0
    diagnostic_clarity: bool = False
    medication_started: bool = False
    discharge_ready: bool = False
    done: bool = False
    last_outcome: str = ""
    retrieved_analogies: List[str] = field(default_factory=list)
    web_augmentation: Dict[str, Any] = field(default_factory=dict)
    task_graph: Dict[str, Any] = field(default_factory=dict)
    context_observation: Dict[str, Any] = field(default_factory=dict)
    message_log: List[str] = field(default_factory=list)
    repeated_actions: int = 0
    last_action_signature: str = ""
    last_consult_target: str = ""


@dataclass
class StepEvaluation:
    milestone_score: float
    coalition_score: float
    safety_score: float
    efficiency_score: float
    terminal_score: float
    task_graph_score: float
    task_graph_loss: float
    task_graph: Dict[str, Any]
    web_augmentation: Dict[str, Any]
    context_observation: Dict[str, Any]
    done: bool
    outcome_text: str
    stakeholder_updates: List[str]
    retrieved_analogies: List[str]


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


def _jaccard_similarity(left: str, right: str) -> float:
    left_tokens = {item for item in _normalize_text(left).split(" ") if item}
    right_tokens = {item for item in _normalize_text(right).split(" ") if item}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


class MIMICCouncilSimulator:
    """Balanced long-horizon simulator built from MIMIC-derived encounter seeds."""

    def __init__(
        self,
        data_root: str | Path,
        max_steps: int = 6,
        sample_size: int = 3000,
        table_row_limit: Optional[int] = None,
        seed: Optional[int] = None,
        llm_search_augmenter: Optional[LLMSearchAugmenter] = None,
        context_manager: Optional[ContextLLMManager] = None,
    ) -> None:
        self.dataset_root = self._resolve_dataset_root(Path(data_root))
        self.data_source = "mimic_bootstrap" if self.dataset_root is not None else "synthetic_bootstrap"
        self.max_steps = max(5, int(max_steps))
        self.sample_size = max(100, int(sample_size))
        self.table_row_limit = (
            max(10_000, self.sample_size * 50)
            if table_row_limit is None
            else max(1_000, int(table_row_limit))
        )
        self._rand = random.Random(seed)
        self.records: List[EncounterRecord] = []
        self.medication_terms: List[str] = []
        self.scenario_buckets: Dict[str, List[EncounterRecord]] = {name: [] for name in SCENARIO_TYPES}
        self.trajectory_archive: List[Dict[str, Any]] = []
        self.llm_search_augmenter = llm_search_augmenter or LLMSearchAugmenter.from_env()
        self.context_manager = context_manager or ContextLLMManager()
        self.scenario_cursor = 0
        if self.dataset_root is None:
            self._load_synthetic_records()
        else:
            self._load_data()

    def _resolve_dataset_root(self, data_root: Path) -> Optional[Path]:
        if not data_root.exists():
            return None
        if (data_root / "hosp").exists() and (data_root / "icu").exists():
            return data_root
        for candidate in data_root.rglob("*"):
            if candidate.is_dir() and (candidate / "hosp").exists() and (candidate / "icu").exists():
                return candidate
        return None

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

        compression: Optional[str] = None
        try:
            with open(path, "rb") as handle:
                compression = "gzip" if handle.read(2) == b"\x1f\x8b" else None
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
            return pd.read_csv(**kwargs, engine="python", on_bad_lines="skip")

    def _read_filtered_csv(
        self,
        rel_path: str,
        filter_col: str,
        filter_values: Sequence[int],
        usecols: Optional[Sequence[str]] = None,
        parse_dates: Optional[Sequence[str]] = None,
        chunksize: int = 200_000,
    ) -> pd.DataFrame:
        values = {_safe_int(value) for value in filter_values}
        path = self.dataset_root / rel_path
        columns = list(usecols) if usecols else []
        if not path.exists() or not values:
            return pd.DataFrame(columns=columns)

        compression: Optional[str] = None
        try:
            with open(path, "rb") as handle:
                compression = "gzip" if handle.read(2) == b"\x1f\x8b" else None
        except OSError:
            compression = None

        chunks: List[pd.DataFrame] = []
        try:
            reader = pd.read_csv(
                path,
                usecols=columns or None,
                parse_dates=list(parse_dates) if parse_dates else None,
                compression=compression,
                low_memory=False,
                chunksize=chunksize,
            )
        except (ParserError, MemoryError):
            reader = pd.read_csv(
                path,
                usecols=columns or None,
                parse_dates=list(parse_dates) if parse_dates else None,
                compression=compression,
                low_memory=False,
                chunksize=chunksize,
                engine="python",
                on_bad_lines="skip",
            )

        for chunk in reader:
            if filter_col not in chunk.columns:
                continue
            filtered = chunk.dropna(subset=[filter_col]).copy()
            if filtered.empty:
                continue
            filtered[filter_col] = filtered[filter_col].map(_safe_int)
            filtered = filtered[filtered[filter_col].isin(values)]
            if not filtered.empty:
                chunks.append(filtered)

        if not chunks:
            return pd.DataFrame(columns=columns)
        return pd.concat(chunks, ignore_index=True)

    def _build_medication_terms(self, med_frames: Sequence[pd.DataFrame]) -> List[str]:
        terms: set[str] = set()
        for frame in med_frames:
            if frame.empty or "medication_text" not in frame.columns:
                continue
            for raw in frame["medication_text"].dropna().astype(str).tolist():
                norm = _normalize_text(raw)
                if len(norm) >= 3:
                    terms.add(norm)
                first = norm.split(" ")[0] if norm else ""
                if len(first) >= 3:
                    terms.add(first)
        return sorted(terms)

    def _top_terms(
        self,
        frame: pd.DataFrame,
        value_col: str,
        group_col: str = "hadm_id",
        limit: int = 4,
    ) -> Dict[int, Tuple[str, ...]]:
        if frame.empty or value_col not in frame.columns or group_col not in frame.columns:
            return {}

        cleaned = frame[[group_col, value_col]].dropna().copy()
        if cleaned.empty:
            return {}
        cleaned[value_col] = cleaned[value_col].astype(str).map(_normalize_text)
        cleaned = cleaned[cleaned[value_col] != ""]
        if cleaned.empty:
            return {}

        output: Dict[int, Tuple[str, ...]] = {}
        for hadm_id, group in cleaned.groupby(group_col):
            values = group[value_col].value_counts().head(limit).index.tolist()
            output[_safe_int(hadm_id)] = tuple(values)
        return output

    def _load_data(self) -> None:
        admissions = self._read_csv(
            "hosp/admissions.csv.gz",
            usecols=["subject_id", "hadm_id", "admittime", "dischtime", "hospital_expire_flag"],
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
        diagnoses = self._read_csv("hosp/diagnoses_icd.csv.gz", usecols=["hadm_id"], nrows=self.table_row_limit)
        procedures = self._read_csv("hosp/procedures_icd.csv.gz", usecols=["hadm_id"], nrows=self.table_row_limit)
        transfers = self._read_csv("hosp/transfers.csv.gz", usecols=["hadm_id"], nrows=self.table_row_limit)
        icustays = self._read_csv("icu/icustays.csv.gz", usecols=["hadm_id"], nrows=self.table_row_limit)
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
        d_labitems = self._read_csv(
            "hosp/d_labitems.csv.gz",
            usecols=["itemid", "label", "category"],
            nrows=self.table_row_limit,
        )
        labevents = self._read_csv(
            "hosp/labevents.csv.gz",
            usecols=[
                "hadm_id",
                "itemid",
                "valuenum",
                "flag",
                "priority",
                "ref_range_lower",
                "ref_range_upper",
            ],
            nrows=max(250_000, self.sample_size * 500, self.table_row_limit * 5),
        )

        admissions = admissions.dropna(subset=["hadm_id", "subject_id"])
        admissions["hadm_id"] = admissions["hadm_id"].astype(int)
        admissions["subject_id"] = admissions["subject_id"].astype(int)

        diag_counts = diagnoses.groupby("hadm_id").size() if not diagnoses.empty else pd.Series(dtype="int64")
        proc_counts = procedures.groupby("hadm_id").size() if not procedures.empty else pd.Series(dtype="int64")
        transfer_counts = transfers.groupby("hadm_id").size() if not transfers.empty else pd.Series(dtype="int64")
        med_frames = [frame for frame in (prescriptions, pharmacy, emar) if not frame.empty]
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

        icu_hadm_ids = set(icustays["hadm_id"].dropna().astype(int).tolist()) if not icustays.empty else set()
        patient_age = {}
        if not patients.empty:
            patient_age = {
                _safe_int(row.subject_id): _safe_float(row.anchor_age)
                for row in patients.itertuples(index=False)
            }

        lab_counts = pd.Series(dtype="int64")
        abnormal_lab_counts = pd.Series(dtype="int64")
        salient_labs_by_hadm: Dict[int, Tuple[str, ...]] = {}
        salient_lab_categories_by_hadm: Dict[int, Tuple[str, ...]] = {}
        if not labevents.empty:
            labevents = labevents.dropna(subset=["hadm_id", "itemid"])
            if not labevents.empty:
                labevents["hadm_id"] = labevents["hadm_id"].astype(int)
                labevents["itemid"] = labevents["itemid"].astype(int)
                lab_items = d_labitems.dropna(subset=["itemid"]).copy() if not d_labitems.empty else pd.DataFrame()
                if not lab_items.empty:
                    lab_items["itemid"] = lab_items["itemid"].astype(int)
                    lab_items = lab_items.drop_duplicates(subset=["itemid"])
                    labevents = labevents.merge(lab_items, on="itemid", how="left")
                else:
                    labevents["label"] = ""
                    labevents["category"] = ""
                labevents["label"] = labevents["label"].fillna("")
                labevents["category"] = labevents["category"].fillna("")
                labevents["flag"] = labevents["flag"].fillna("").astype(str)
                lab_counts = labevents.groupby("hadm_id").size()
                abnormal_labs = labevents[labevents["flag"].str.strip() != ""]
                abnormal_lab_counts = (
                    abnormal_labs.groupby("hadm_id").size() if not abnormal_labs.empty else pd.Series(dtype="int64")
                )
                salient_labs_by_hadm = self._top_terms(
                    abnormal_labs if not abnormal_labs.empty else labevents,
                    "label",
                )
                salient_lab_categories_by_hadm = self._top_terms(
                    abnormal_labs if not abnormal_labs.empty else labevents,
                    "category",
                )

        self.medication_terms = self._build_medication_terms(med_frames)

        if len(admissions) > self.sample_size:
            lab_supported = admissions[admissions["hadm_id"].isin(set(lab_counts.index.tolist()))]
            sample_parts = []
            if not lab_supported.empty:
                target_lab_rows = min(len(lab_supported), max(1, int(self.sample_size * 0.5)))
                sample_parts.append(lab_supported.sample(n=target_lab_rows, random_state=42))
            sampled_ids = set()
            if sample_parts:
                sampled_ids = set(pd.concat(sample_parts)["hadm_id"].astype(int).tolist())
            remaining_pool = admissions[~admissions["hadm_id"].isin(sampled_ids)]
            remaining_needed = self.sample_size - sum(len(part) for part in sample_parts)
            if remaining_needed > 0:
                sample_parts.append(
                    remaining_pool.sample(n=min(remaining_needed, len(remaining_pool)), random_state=43)
                )
            admissions = pd.concat(sample_parts, ignore_index=True).head(self.sample_size)

        records: List[EncounterRecord] = []
        for row in admissions.itertuples(index=False):
            hadm_id = _safe_int(row.hadm_id)
            subject_id = _safe_int(row.subject_id)
            los_hours = 24.0
            if pd.notna(row.admittime) and pd.notna(row.dischtime):
                delta = row.dischtime - row.admittime
                los_hours = max(delta.total_seconds() / 3600.0, 0.5)

            meds = tuple(str(item) for item in meds_by_hadm.get(hadm_id, [])[:5])
            record = EncounterRecord(
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
                lab_event_count=int(lab_counts.get(hadm_id, 0)),
                abnormal_lab_event_count=int(abnormal_lab_counts.get(hadm_id, 0)),
                salient_labs=salient_labs_by_hadm.get(hadm_id, ()),
                salient_lab_categories=salient_lab_categories_by_hadm.get(hadm_id, ()),
            )
            records.append(record)
        if not records:
            raise ValueError("No encounter records could be built from MIMIC data.")
        self.records = records
        self._bucket_records()

    def _load_synthetic_records(self) -> None:
        self.records = [
            EncounterRecord(
                hadm_id=900001,
                subject_id=910001,
                los_hours=42.0,
                in_icu=1,
                expired_flag=0,
                age=71.0,
                diag_count=0,
                med_count=0,
                proc_count=2,
                transfer_count=2,
                meds=(),
                lab_event_count=11,
                abnormal_lab_event_count=4,
                salient_labs=("troponin", "lactate", "creatinine"),
                salient_lab_categories=("chemistry", "blood gas"),
            ),
            EncounterRecord(
                hadm_id=900002,
                subject_id=910002,
                los_hours=58.0,
                in_icu=0,
                expired_flag=0,
                age=64.0,
                diag_count=1,
                med_count=3,
                proc_count=1,
                transfer_count=1,
                meds=("heparin", "ceftriaxone", "furosemide"),
                lab_event_count=8,
                abnormal_lab_event_count=2,
                salient_labs=("white blood cell count", "sodium"),
                salient_lab_categories=("hematology", "chemistry"),
            ),
            EncounterRecord(
                hadm_id=900003,
                subject_id=910003,
                los_hours=30.0,
                in_icu=0,
                expired_flag=0,
                age=52.0,
                diag_count=2,
                med_count=0,
                proc_count=0,
                transfer_count=0,
                meds=(),
                lab_event_count=4,
                abnormal_lab_event_count=1,
                salient_labs=("potassium",),
                salient_lab_categories=("chemistry",),
            ),
            EncounterRecord(
                hadm_id=900004,
                subject_id=910004,
                los_hours=102.0,
                in_icu=0,
                expired_flag=0,
                age=47.0,
                diag_count=3,
                med_count=1,
                proc_count=0,
                transfer_count=1,
                meds=("acetaminophen",),
                lab_event_count=5,
                abnormal_lab_event_count=0,
                salient_labs=("hemoglobin",),
                salient_lab_categories=("hematology",),
            ),
            EncounterRecord(
                hadm_id=900005,
                subject_id=910005,
                los_hours=68.0,
                in_icu=1,
                expired_flag=0,
                age=79.0,
                diag_count=0,
                med_count=1,
                proc_count=2,
                transfer_count=3,
                meds=("vancomycin",),
                lab_event_count=15,
                abnormal_lab_event_count=5,
                salient_labs=("lactate", "blood culture", "creatinine"),
                salient_lab_categories=("chemistry", "microbiology"),
            ),
            EncounterRecord(
                hadm_id=900006,
                subject_id=910006,
                los_hours=77.0,
                in_icu=0,
                expired_flag=0,
                age=60.0,
                diag_count=2,
                med_count=4,
                proc_count=1,
                transfer_count=2,
                meds=("piperacillin tazobactam", "insulin", "metoprolol"),
                lab_event_count=9,
                abnormal_lab_event_count=2,
                salient_labs=("glucose", "magnesium"),
                salient_lab_categories=("chemistry",),
            ),
            EncounterRecord(
                hadm_id=900007,
                subject_id=910007,
                los_hours=50.0,
                in_icu=0,
                expired_flag=0,
                age=69.0,
                diag_count=1,
                med_count=0,
                proc_count=0,
                transfer_count=1,
                meds=(),
                lab_event_count=3,
                abnormal_lab_event_count=0,
                salient_labs=("platelet count",),
                salient_lab_categories=("hematology",),
            ),
            EncounterRecord(
                hadm_id=900008,
                subject_id=910008,
                los_hours=95.0,
                in_icu=0,
                expired_flag=0,
                age=56.0,
                diag_count=2,
                med_count=1,
                proc_count=0,
                transfer_count=2,
                meds=("amoxicillin",),
                lab_event_count=6,
                abnormal_lab_event_count=1,
                salient_labs=("bilirubin",),
                salient_lab_categories=("chemistry",),
            ),
        ]
        self.medication_terms = sorted(
            {
                normalized
                for record in self.records
                for med in record.meds
                for normalized in [_normalize_text(med)]
                if normalized
            }
        )
        self._bucket_records()

    def _bucket_records(self) -> None:
        for record in self.records:
            self.scenario_buckets[self._classify_record(record)].append(record)
        for scenario in SCENARIO_TYPES:
            if not self.scenario_buckets[scenario]:
                ranked = sorted(
                    self.records,
                    key=lambda item: self._scenario_affinity(scenario, item),
                    reverse=True,
                )
                self.scenario_buckets[scenario] = ranked[: max(10, min(len(ranked), 100))]

    def _scenario_affinity(self, scenario_type: str, record: EncounterRecord) -> float:
        if scenario_type == "medication_alignment":
            return float(record.med_count) + 0.2 * float(record.abnormal_lab_event_count)
        if scenario_type == "discharge_negotiation":
            return (
                2.0 * float(record.expired_flag == 0 and record.in_icu == 0)
                + 0.5 * float(record.los_hours >= 48)
                - 0.3 * float(record.abnormal_lab_event_count)
            )
        if scenario_type == "conservative_monitoring":
            return (
                1.5 * float(record.med_count == 0)
                + 1.0 * float(record.abnormal_lab_event_count <= 1)
                + 0.5 * float(record.diag_count > 0)
            )
        return (
            1.5 * float(record.diag_count == 0)
            + 1.0 * float(record.abnormal_lab_event_count >= 2)
            + 0.5 * float(record.proc_count > 0)
        )

    def _classify_record(self, record: EncounterRecord) -> str:
        if record.abnormal_lab_event_count >= 3 and record.diag_count <= 1:
            return "diagnostic_ambiguity"
        if record.med_count > 0:
            return "medication_alignment"
        if record.expired_flag == 0 and record.in_icu == 0 and record.los_hours >= 48:
            return "discharge_negotiation"
        if record.diag_count == 0 or record.proc_count > 0:
            return "diagnostic_ambiguity"
        return "conservative_monitoring"

    def _late_discharge_ready(self, record: EncounterRecord) -> bool:
        return (
            record.expired_flag == 0
            and record.in_icu == 0
            and record.abnormal_lab_event_count <= 1
            and record.los_hours >= 48
        )

    def _ground_truth_for_step(
        self,
        scenario_type: str,
        record: EncounterRecord,
        step_idx: int,
    ) -> str:
        late_phase = step_idx >= max(0, self.max_steps - 2)
        if scenario_type == "medication_alignment":
            if step_idx == 0:
                return "diagnosis" if (record.diag_count == 0 or record.abnormal_lab_event_count >= 2) else "medication"
            if late_phase and self._late_discharge_ready(record):
                return "discharge"
            return "medication"

        if scenario_type == "conservative_monitoring":
            if step_idx == 0 and (record.diag_count == 0 or record.abnormal_lab_event_count >= 3):
                return "diagnosis"
            if late_phase and self._late_discharge_ready(record):
                return "discharge"
            return "no_action"

        if scenario_type == "discharge_negotiation":
            if step_idx == 0 and record.abnormal_lab_event_count >= 3:
                return "diagnosis"
            if late_phase:
                return "discharge" if self._late_discharge_ready(record) else "no_action"
            return "no_action"

        if step_idx <= 2:
            return "diagnosis"
        if late_phase:
            return "discharge" if self._late_discharge_ready(record) else "no_action"
        return "no_action"

    def _stage_expected_category(
        self,
        scenario_type: str,
        expected_action_type: str,
        base_category: str,
        record: EncounterRecord,
    ) -> str:
        if base_category != "no_action":
            return base_category
        if expected_action_type == "propose":
            if scenario_type == "medication_alignment":
                return "medication"
            if scenario_type == "discharge_negotiation":
                return "discharge"
            if scenario_type in {"diagnostic_ambiguity", "conservative_monitoring"}:
                return "diagnosis"
        if expected_action_type == "commit":
            if scenario_type == "medication_alignment":
                return "medication"
            if scenario_type == "discharge_negotiation" or self._late_discharge_ready(record):
                return "discharge"
            return "diagnosis"
        if expected_action_type == "resolve" and scenario_type == "discharge_negotiation":
            return "discharge"
        return base_category

    def _sample_record(self, scenario_type: Optional[str] = None) -> Tuple[str, EncounterRecord]:
        if scenario_type is None:
            scenario_type = SCENARIO_TYPES[self.scenario_cursor % len(SCENARIO_TYPES)]
            self.scenario_cursor += 1
        bucket = self.scenario_buckets.get(scenario_type, self.records)
        return scenario_type, self._rand.choice(bucket)

    def _patient_snapshot(self, record: EncounterRecord) -> Dict[str, Any]:
        return {
            "data_source": self.data_source,
            "age_band": "older_adult" if record.age >= 65 else "adult",
            "los_hours": round(record.los_hours, 1),
            "icu": bool(record.in_icu),
            "diagnosis_signal_count": record.diag_count,
            "treatment_signal_count": record.med_count,
            "procedure_signal_count": record.proc_count,
            "transfer_signal_count": record.transfer_count,
            "lab_signal_count": record.lab_event_count,
            "abnormal_lab_signal_count": record.abnormal_lab_event_count,
            "candidate_medications": [_normalize_text(item) for item in record.meds if _normalize_text(item)][:3],
            "salient_labs": list(record.salient_labs[:4]),
            "salient_lab_categories": list(record.salient_lab_categories[:4]),
        }

    def _mission_brief(self, scenario_type: str, record: EncounterRecord) -> str:
        lab_hint = ""
        if record.abnormal_lab_event_count > 0 and record.salient_labs:
            lab_hint = f" Recent lab abnormalities include {', '.join(record.salient_labs[:2])}."
        if scenario_type == "medication_alignment":
            return (
                "Multiple teams think the patient may need active treatment, but the council has to "
                "decide what to start, when to start it, and how to keep the pharmacist aligned."
                f"{lab_hint}"
            )
        if scenario_type == "discharge_negotiation":
            return (
                "Clinical pressure is easing, but operations, family expectations, and transition risk "
                "are not fully aligned yet."
                f"{lab_hint}"
            )
        if scenario_type == "conservative_monitoring":
            return (
                "The case is stable enough to avoid overreacting, but everyone is watching for signs "
                "that conservative management might no longer be enough."
                f"{lab_hint}"
            )
        return (
            "The council is still piecing together what is really going on. Early confidence may be "
            "misleading, and the team needs the right information before making irreversible moves."
            f"{lab_hint}"
        )

    def _long_horizon_goals(self, scenario_type: str) -> List[str]:
        goals = [
            "Build enough coalition support that stakeholders stop working at cross-purposes.",
            "Pick actions that match the hidden clinical trajectory instead of chasing superficial cues.",
            "Preserve safety even when throughput or family pressure increases.",
        ]
        if scenario_type == "discharge_negotiation":
            goals.append("Land a safe transition plan without triggering avoidable conflict.")
        elif scenario_type == "medication_alignment":
            goals.append("Get treatment moving without breaking medication safety constraints.")
        else:
            goals.append("Avoid premature escalation while keeping the episode moving forward.")
        return goals

    def _initial_conflicts(self, scenario_type: str, record: EncounterRecord) -> List[str]:
        conflicts = []
        if scenario_type == "discharge_negotiation":
            conflicts.append("Bed manager wants flow, family liaison wants confidence before discharge.")
        if scenario_type == "medication_alignment":
            conflicts.append("Pharmacist wants specificity before treatment proceeds.")
        if scenario_type == "diagnostic_ambiguity":
            conflicts.append("Attending physician and triage nurse disagree on how much uncertainty remains.")
        if record.abnormal_lab_event_count > 0 and record.salient_labs:
            conflicts.append(
                f"Lab signals ({', '.join(record.salient_labs[:2])}) raise uncertainty about whether the current plan is still valid."
            )
        if record.transfer_count > 1:
            conflicts.append("Frequent handoffs created fragmented context across teams.")
        return conflicts[:3]

    def _build_stages(self, scenario_type: str, record: EncounterRecord) -> List[CouncilStage]:
        stages: List[CouncilStage] = []
        for step_idx in range(self.max_steps):
            ground_truth = self._ground_truth_for_step(scenario_type, record, step_idx)
            if step_idx == 0:
                expected_action_type = "consult"
                stages.append(
                    CouncilStage(
                        step_index=step_idx,
                        phase_name="sensemaking",
                        expected_action_type=expected_action_type,
                        expected_category=self._stage_expected_category(
                            scenario_type,
                            expected_action_type,
                            ground_truth,
                            record,
                        ),
                        preferred_targets=(PRIMARY_TARGETS[scenario_type],),
                        rationale="Open by consulting the stakeholder most likely to reduce uncertainty.",
                    )
                )
            elif step_idx == 1:
                expected_action_type = "propose"
                stages.append(
                    CouncilStage(
                        step_index=step_idx,
                        phase_name="alignment",
                        expected_action_type=expected_action_type,
                        expected_category=self._stage_expected_category(
                            scenario_type,
                            expected_action_type,
                            ground_truth,
                            record,
                        ),
                        preferred_targets=(PRIMARY_TARGETS[scenario_type], SECONDARY_TARGETS[scenario_type]),
                        rationale="Turn signals into an explicit directional plan the council can react to.",
                    )
                )
            elif step_idx == 2:
                expected_action_type = "commit"
                stages.append(
                    CouncilStage(
                        step_index=step_idx,
                        phase_name="execution",
                        expected_action_type=expected_action_type,
                        expected_category=self._stage_expected_category(
                            scenario_type,
                            expected_action_type,
                            ground_truth,
                            record,
                        ),
                        preferred_targets=(PRIMARY_TARGETS[scenario_type],),
                        rationale="Make the first concrete move once the council has enough direction.",
                    )
                )
            elif step_idx == self.max_steps - 2:
                expected_action_type = "resolve"
                stages.append(
                    CouncilStage(
                        step_index=step_idx,
                        phase_name="conflict_resolution",
                        expected_action_type=expected_action_type,
                        expected_category=self._stage_expected_category(
                            scenario_type,
                            expected_action_type,
                            ground_truth,
                            record,
                        ),
                        preferred_targets=(SECONDARY_TARGETS[scenario_type], "family_liaison"),
                        rationale="Late-stage friction needs to be managed so the final action can land cleanly.",
                    )
                )
            elif step_idx == self.max_steps - 1:
                expected_action_type = "commit"
                stages.append(
                    CouncilStage(
                        step_index=step_idx,
                        phase_name="handoff",
                        expected_action_type=expected_action_type,
                        expected_category=self._stage_expected_category(
                            scenario_type,
                            expected_action_type,
                            ground_truth,
                            record,
                        ),
                        preferred_targets=("bed_manager", "family_liaison"),
                        rationale="Finish the episode with a decisive long-horizon handoff move.",
                    )
                )
            else:
                expected_action_type = "delegate"
                stages.append(
                    CouncilStage(
                        step_index=step_idx,
                        phase_name="coordination",
                        expected_action_type=expected_action_type,
                        expected_category=self._stage_expected_category(
                            scenario_type,
                            expected_action_type,
                            ground_truth,
                            record,
                        ),
                        preferred_targets=(SECONDARY_TARGETS[scenario_type],),
                        rationale="Keep the right stakeholder looped in while the plan unfolds.",
                    )
                )
        return stages

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario_type: Optional[str] = None,
        difficulty: str = "medium",
    ) -> EpisodeSnapshot:
        if seed is not None:
            self._rand.seed(seed)
        difficulty = difficulty if difficulty in ("easy", "medium", "hard") else "medium"
        scenario_type, record = self._sample_record(scenario_type)
        stakeholders = {
            name: StakeholderState(
                name=name,
                alignment=0.45 if name != PRIMARY_TARGETS[scenario_type] else 0.55,
                influence=0.9 if name in ("attending_physician", "bed_manager") else 0.7,
            )
            for name in STAKEHOLDER_DESCRIPTIONS
        }
        snapshot = EpisodeSnapshot(
            scenario_id=episode_id or f"{scenario_type}-{record.hadm_id}-{self._rand.randint(1000, 9999)}",
            scenario_type=scenario_type,
            difficulty=difficulty,
            record=record,
            mission_brief=self._mission_brief(scenario_type, record),
            long_horizon_goals=self._long_horizon_goals(scenario_type),
            visible_conflicts=self._initial_conflicts(scenario_type, record),
            patient_snapshot=self._patient_snapshot(record),
            stages=self._build_stages(scenario_type, record),
            stakeholders=stakeholders,
        )
        snapshot.message_log.append(
            "Council opened. Stakeholders are present but not fully aligned on the next move."
        )
        return snapshot

    def _medication_match(self, selected_medication: str, record: EncounterRecord) -> bool:
        selected = _normalize_text(selected_medication)
        if not selected:
            return False
        if selected in self.medication_terms:
            return True
        candidates = [_normalize_text(item) for item in record.meds if _normalize_text(item)]
        return any(selected in candidate or candidate in selected for candidate in candidates)

    def _retrieve_analogies(
        self,
        snapshot: EpisodeSnapshot,
        action_signature: str,
        limit: int = 3,
    ) -> List[str]:
        ranked: List[Tuple[float, str]] = []
        for item in self.trajectory_archive:
            score = 0.0
            score += 0.45 if item.get("scenario_type") == snapshot.scenario_type else 0.0
            score += 0.20 if item.get("expected_action_type") == snapshot.stages[min(snapshot.step_count, len(snapshot.stages) - 1)].expected_action_type else 0.0
            score += 0.20 if item.get("category") == item.get("expected_category") else 0.0
            score += 0.15 * _jaccard_similarity(action_signature, str(item.get("summary", "")))
            if score <= 0.0:
                continue
            ranked.append((score, str(item.get("summary", ""))))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [text for _, text in ranked[:limit]]

    def _expected_action_payload(
        self,
        stage: CouncilStage,
        scenario_type: str,
        patient_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        medications = list(patient_snapshot.get("candidate_medications", []))
        category = stage.expected_category
        medication = medications[0] if category == "medication" and medications else ""
        target = stage.preferred_targets[0] if stage.preferred_targets else ""
        if stage.expected_action_type == "commit":
            target = ""
        return {
            "action_type": stage.expected_action_type,
            "category": category,
            "target": target,
            "medication": medication,
            "message": stage.rationale,
            "scenario_type": scenario_type,
            "phase_name": stage.phase_name,
        }

    def _validate_action(self, action: Dict[str, Any], stage: CouncilStage) -> Tuple[bool, List[str]]:
        action_type = str(action.get("action_type", "") or "").strip()
        target = str(action.get("target", "") or "").strip()
        category = str(action.get("category", "") or "").strip()
        reasons: List[str] = []

        if action_type in {"consult", "delegate", "resolve"} and not target:
            reasons.append(f"{action_type} requires a target")
        if action_type in {"propose", "commit"} and not category:
            reasons.append(f"{action_type} requires a category")
        if action_type == "consult" and category:
            reasons.append("consult should not set a category")
        if action_type in {"propose", "commit"} and target:
            reasons.append(f"{action_type} does not use a target in this environment")
        if action_type not in {"consult", "propose", "delegate", "resolve", "commit"}:
            reasons.append("unknown action type")
        return (len(reasons) == 0, reasons)

    def _build_task_graph(
        self,
        snapshot: EpisodeSnapshot,
        stage: CouncilStage,
        action: Dict[str, Any],
        task_graph_loss: float,
        task_graph_score: float,
        active_step_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        active_index = (
            min(max(0, int(active_step_index)), len(snapshot.stages) - 1)
            if active_step_index is not None
            else snapshot.step_count
        )
        nodes = []
        for item in snapshot.stages:
            if item.step_index < active_index:
                status = "complete"
            elif item.step_index == active_index:
                status = "active"
            else:
                status = "pending"
            nodes.append(
                {
                    "id": f"step_{item.step_index}_{item.phase_name}",
                    "status": status,
                    "phase": item.phase_name,
                    "expected_action_type": item.expected_action_type,
                    "expected_category": item.expected_category,
                    "preferred_targets": list(item.preferred_targets),
                }
            )
        return {
            "active_node": f"step_{snapshot.stages[active_index].step_index}_{snapshot.stages[active_index].phase_name}",
            "loss": round(task_graph_loss, 4),
            "score": round(task_graph_score, 4),
            "nodes": nodes,
            "last_action": {
                "action_type": str(action.get("action_type", "") or ""),
                "target": str(action.get("target", "") or ""),
                "category": str(action.get("category", "") or ""),
                "medication": str(action.get("medication", "") or ""),
            },
        }

    def _task_graph_loss(
        self,
        *,
        action_type: str,
        category: str,
        stage: CouncilStage,
        action_type_match: bool,
        category_match: bool,
        target_match: bool,
        medication_match: bool,
        safety_score: float,
        visible_conflicts: Sequence[str],
        action_valid: bool,
    ) -> float:
        phase_loss = 0.0 if action_type_match else 1.0
        if category_match:
            category_loss = 0.0
        elif action_type == "consult" and not category:
            category_loss = 0.35
        else:
            category_loss = 1.0
        target_loss = 0.0 if target_match else 1.0
        evidence_loss = 0.0 if medication_match else 1.0
        safety_loss = 1.0 - safety_score
        conflict_pressure = min(1.0, len(visible_conflicts) / 3.0)
        if action_type in ("resolve", "consult"):
            conflict_loss = max(0.0, conflict_pressure - 0.35)
        else:
            conflict_loss = conflict_pressure
        loss = (
            0.30 * phase_loss
            + 0.25 * category_loss
            + 0.15 * target_loss
            + 0.10 * evidence_loss
            + 0.15 * safety_loss
            + 0.05 * conflict_loss
        )
        if not action_valid:
            loss += 0.20
        return max(0.0, min(1.0, loss))

    def _build_web_augmentation(
        self,
        snapshot: EpisodeSnapshot,
        stage: CouncilStage,
        action: Dict[str, Any],
    ) -> Dict[str, Any]:
        next_stage = snapshot.stages[min(snapshot.step_count + 1, len(snapshot.stages) - 1)]
        context = self.llm_search_augmenter.augment(
            mission_brief=snapshot.mission_brief,
            scenario_type=snapshot.scenario_type,
            phase_name=stage.phase_name,
            stage_rationale=stage.rationale,
            action=action,
            patient_snapshot=snapshot.patient_snapshot,
            expected_action=self._expected_action_payload(stage, snapshot.scenario_type, snapshot.patient_snapshot),
            trajectory_archive=self.trajectory_archive,
        )
        context["next_expected_action"] = self._expected_action_payload(
            next_stage,
            snapshot.scenario_type,
            snapshot.patient_snapshot,
        )
        context["generated_at_ms"] = timestamp_ms()
        return context

    def _stakeholder_note(
        self,
        snapshot: EpisodeSnapshot,
        target: str,
        expected_category: str,
    ) -> str:
        if target == "attending_physician":
            return (
                "Attending: I want a cleaner theory of the case before we pretend the uncertainty is gone."
                if expected_category == "diagnosis"
                else "Attending: The plan should stop drifting and turn into a concrete clinical direction."
            )
        if target == "triage_nurse":
            return (
                "Triage nurse: Bedside signals still feel noisy; premature certainty would be risky."
                if expected_category == "diagnosis"
                else "Triage nurse: If we hold steady, we need a clear reason for not escalating."
            )
        if target == "pharmacist":
            return (
                "Pharmacist: If treatment starts, it needs to be specific enough to defend."
                if expected_category == "medication"
                else "Pharmacist: I can support the plan, but not if medication logic stays vague."
            )
        if target == "bed_manager":
            return (
                "Bed manager: Flow matters, but a rushed transition will bounce back on us."
                if expected_category == "discharge"
                else "Bed manager: I need to know whether this patient is staying put or moving soon."
            )
        return (
            "Family liaison: The family can handle uncertainty if we communicate a coherent next step."
        )

    def evaluate_step(self, snapshot: EpisodeSnapshot, action: Dict[str, Any]) -> StepEvaluation:
        stage = snapshot.stages[min(snapshot.step_count, len(snapshot.stages) - 1)]
        next_stage = snapshot.stages[min(snapshot.step_count + 1, len(snapshot.stages) - 1)]
        action_type = str(action.get("action_type", "")).strip()
        target = str(action.get("target") or "").strip()
        category = str(action.get("category") or "").strip()
        medication = str(action.get("medication") or "").strip()
        message = str(action.get("message") or "").strip()
        action_signature = f"{action_type}|{target}|{category}|{medication}|{message}"
        action_valid, invalid_reasons = self._validate_action(action, stage)
        previous_consult_target = snapshot.last_consult_target

        action_type_match = action_type == stage.expected_action_type
        if stage.expected_category == "no_action":
            category_match = True
        else:
            category_match = category == stage.expected_category if category else False
        if action_type in {"propose", "commit"}:
            target_match = True
        else:
            target_match = not stage.preferred_targets or target in stage.preferred_targets
        medication_match = (
            stage.expected_category != "medication"
            or self._medication_match(medication or message, snapshot.record)
        )

        milestone_score = 0.15
        if action_type_match:
            milestone_score += 0.35
        if category_match:
            milestone_score += 0.30
        if target_match:
            milestone_score += 0.10
        if medication_match:
            milestone_score += 0.10
        if not action_valid:
            milestone_score -= 0.25
        milestone_score = max(0.0, min(1.0, milestone_score))

        stakeholder_updates: List[str] = []
        if not action_valid:
            stakeholder_updates.append(f"Validation: {'; '.join(invalid_reasons)}.")

        if action_type == "consult" and action_valid and target in snapshot.stakeholders:
            stakeholder = snapshot.stakeholders[target]
            stakeholder.consulted += 1
            stakeholder.alignment = min(1.0, stakeholder.alignment + (0.14 if target_match else 0.04))
            stakeholder.last_note = self._stakeholder_note(snapshot, target, stage.expected_category)
            stakeholder_updates.append(f"{target}: {stakeholder.last_note}")
            if target not in snapshot.patient_snapshot.get("consulted_stakeholders", []):
                snapshot.patient_snapshot.setdefault("consulted_stakeholders", []).append(target)
            if target in ("attending_physician", "triage_nurse"):
                snapshot.diagnostic_clarity = snapshot.diagnostic_clarity or stage.expected_category == "diagnosis"
            if previous_consult_target == target:
                stakeholder_updates.append("System: Repeated consult on the same stakeholder added less new information.")

        if action_type in ("propose", "delegate", "resolve", "commit") and action_valid:
            primary = snapshot.stakeholders[PRIMARY_TARGETS[snapshot.scenario_type]]
            secondary = snapshot.stakeholders[SECONDARY_TARGETS[snapshot.scenario_type]]
            if category_match:
                primary.alignment = min(1.0, primary.alignment + 0.12)
                secondary.alignment = min(1.0, secondary.alignment + 0.08)
            else:
                primary.alignment = max(0.0, primary.alignment - 0.10)
                secondary.alignment = max(0.0, secondary.alignment - 0.05)

        if action_type == "resolve" and action_valid and snapshot.visible_conflicts:
            snapshot.visible_conflicts = snapshot.visible_conflicts[1:]
            stakeholder_updates.append("System: One visible conflict eased after the coordinator addressed it directly.")

        if action_type == "commit" and action_valid and category == "medication" and category_match and medication_match:
            snapshot.medication_started = True
            stakeholder_updates.append("System: Treatment was started and the council moved from debate to action.")
        if action_type == "commit" and action_valid and category == "discharge" and category_match:
            snapshot.discharge_ready = True
            stakeholder_updates.append("System: Transition planning moved into a real handoff state.")
        if action_type in ("propose", "commit") and action_valid and category == "diagnosis" and category_match:
            snapshot.diagnostic_clarity = True

        coalition_values = [item.alignment for item in snapshot.stakeholders.values()]
        coalition_score = sum(coalition_values) / max(1, len(coalition_values))

        safety_score = 1.0
        if not action_valid:
            safety_score -= 0.15
        if action_type == "commit" and category == "discharge" and stage.expected_category != "discharge":
            safety_score -= 0.65
        if action_type == "commit" and category == "medication" and not snapshot.diagnostic_clarity and snapshot.scenario_type == "diagnostic_ambiguity":
            safety_score -= 0.35
        if action_type == "commit" and category == "medication" and not medication_match:
            safety_score -= 0.20
        safety_score = max(0.0, min(1.0, safety_score))

        task_graph_loss = self._task_graph_loss(
            action_type=action_type,
            category=category,
            stage=stage,
            action_type_match=action_type_match,
            category_match=category_match,
            target_match=target_match,
            medication_match=medication_match,
            safety_score=safety_score,
            visible_conflicts=snapshot.visible_conflicts,
            action_valid=action_valid,
        )
        task_graph_score = 1.0 - task_graph_loss
        task_graph = self._build_task_graph(
            snapshot=snapshot,
            stage=stage,
            action=action,
            task_graph_loss=task_graph_loss,
            task_graph_score=task_graph_score,
            active_step_index=min(snapshot.step_count + 1, len(snapshot.stages) - 1),
        )
        web_augmentation = self._build_web_augmentation(snapshot, stage, action)

        if snapshot.last_action_signature == action_signature and action_signature:
            snapshot.repeated_actions += 1
        else:
            snapshot.repeated_actions = 0
        snapshot.last_action_signature = action_signature
        efficiency_score = max(0.0, 1.0 - 0.12 * snapshot.repeated_actions)
        if action_type == "consult" and target and previous_consult_target == target and snapshot.repeated_actions > 0:
            efficiency_score = max(0.0, efficiency_score - 0.10)
        if action_type == "consult" and action_valid:
            snapshot.last_consult_target = target

        done = snapshot.step_count + 1 >= self.max_steps
        if action_valid and action_type == "commit" and category == "discharge" and stage.expected_category == "discharge":
            done = True

        terminal_score = 0.0
        if done:
            coalition_gate = coalition_score >= 0.58
            if action_type == "commit" and stage.expected_category != "no_action":
                action_gate = category == stage.expected_category
            else:
                action_gate = action_type_match
            terminal_score = 1.0 if coalition_gate and action_gate and safety_score >= 0.7 else 0.25

        if milestone_score >= 0.85:
            outcome_text = f"High-value move. The council handled the {stage.phase_name} phase with strong alignment."
        elif milestone_score >= 0.55:
            outcome_text = f"Usable move. The phase advanced, but the council still has drift to clean up."
        else:
            outcome_text = f"Weak move. The council created drag instead of solving the current {stage.phase_name} task."

        retrieved_analogies = []
        if milestone_score < 0.55:
            retrieved_analogies = self._retrieve_analogies(snapshot, action_signature)
            if retrieved_analogies:
                stakeholder_updates.append("Memory: Similar prior episodes suggest the council is drifting off the strong path.")
        context_observation = self.context_manager.build_observation(
            action=action,
            expected_action=self._expected_action_payload(stage, snapshot.scenario_type, snapshot.patient_snapshot),
            next_expected_action=self._expected_action_payload(
                next_stage,
                snapshot.scenario_type,
                snapshot.patient_snapshot,
            ),
            retrieved_trajectories=web_augmentation.get("trajectory_alignments", []),
            llm_search=web_augmentation,
            task_graph=task_graph,
            task_graph_loss=task_graph_loss,
            milestone_score=milestone_score,
            safety_score=safety_score,
            coalition_score=coalition_score,
            patient_snapshot=snapshot.patient_snapshot,
        )
        if web_augmentation.get("valid_use_cases"):
            top_case = web_augmentation["valid_use_cases"][0]
            stakeholder_updates.append(
                f"LLM search: {top_case.get('case', 'current action')} support={top_case.get('support', 0.0)}."
            )
        stakeholder_updates.append(
            "Guidance: "
            f"{context_observation.get('next_step_guidance', 'retain')} -> "
            f"{context_observation.get('correction_signal', {}).get('rationale', '')}"
        )

        archive_summary = (
            f"{snapshot.scenario_type} step {snapshot.step_count}: expected {stage.expected_action_type}/"
            f"{stage.expected_category}, saw {action_type or 'none'}/{category or 'none'}."
        )
        self.trajectory_archive.append(
            {
                "scenario_type": snapshot.scenario_type,
                "expected_action_type": stage.expected_action_type,
                "expected_category": stage.expected_category,
                "category": category,
                "summary": archive_summary,
                "task_graph_loss": round(task_graph_loss, 4),
                "semantic_overlap": context_observation.get("confidence", 0.0),
            }
        )

        return StepEvaluation(
            milestone_score=milestone_score,
            coalition_score=coalition_score,
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            terminal_score=terminal_score,
            task_graph_score=task_graph_score,
            task_graph_loss=task_graph_loss,
            task_graph=task_graph,
            web_augmentation=web_augmentation,
            context_observation=context_observation,
            done=done,
            outcome_text=outcome_text,
            stakeholder_updates=stakeholder_updates,
            retrieved_analogies=retrieved_analogies,
        )

    def advance(self, snapshot: EpisodeSnapshot, action: Dict[str, Any]) -> Dict[str, Any]:
        evaluation = self.evaluate_step(snapshot, action)
        snapshot.step_count += 1
        snapshot.done = evaluation.done
        snapshot.last_outcome = evaluation.outcome_text
        snapshot.retrieved_analogies = evaluation.retrieved_analogies
        snapshot.web_augmentation = evaluation.web_augmentation
        snapshot.task_graph = evaluation.task_graph
        snapshot.context_observation = evaluation.context_observation
        snapshot.message_log.extend(evaluation.stakeholder_updates)
        return {
            "evaluation": evaluation,
            "scoreboard": {
                "milestone": round(evaluation.milestone_score, 4),
                "coalition": round(evaluation.coalition_score, 4),
                "safety": round(evaluation.safety_score, 4),
                "efficiency": round(evaluation.efficiency_score, 4),
                "terminal": round(evaluation.terminal_score, 4),
                "task_graph": round(evaluation.task_graph_score, 4),
                "task_graph_loss": round(evaluation.task_graph_loss, 4),
                "web_evidence_count": float(evaluation.web_augmentation.get("evidence_count", 0)),
                "context_confidence": float(evaluation.context_observation.get("confidence", 0.0)),
                "progress": round(snapshot.step_count / max(1, self.max_steps), 4),
            },
        }

    def export_state(self, snapshot: EpisodeSnapshot) -> Dict[str, Any]:
        return {
            "scenario_id": snapshot.scenario_id,
            "scenario_type": snapshot.scenario_type,
            "difficulty": snapshot.difficulty,
            "step_count": snapshot.step_count,
            "max_steps": self.max_steps,
            "coalition_support": {
                name: round(stakeholder.alignment, 4)
                for name, stakeholder in snapshot.stakeholders.items()
            },
            "consulted_stakeholders": [
                name for name, stakeholder in snapshot.stakeholders.items() if stakeholder.consulted > 0
            ],
            "diagnostic_clarity": snapshot.diagnostic_clarity,
            "medication_started": snapshot.medication_started,
            "discharge_ready": snapshot.discharge_ready,
            "hidden_targets": [asdict(stage) for stage in snapshot.stages],
            "task_graph": snapshot.task_graph,
            "web_augmentation": snapshot.web_augmentation,
            "context_observation": snapshot.context_observation,
            "archived_trajectory_size": len(self.trajectory_archive),
        }
