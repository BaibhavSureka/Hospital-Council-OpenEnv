"""Optional web-search augmentation for Hospital Council trajectories."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence
from urllib import error, request


def _normalize_text(value: str) -> str:
    return " ".join(str(value).lower().replace("_", " ").split())


def _tokens(value: str) -> set[str]:
    return {token for token in _normalize_text(value).split(" ") if len(token) >= 3}


def _overlap(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


@dataclass
class WebSearchAugmenter:
    """Retrieves and compresses external evidence for a query-action pair.

    The augmenter uses Serper when ``SERPER_API_KEY`` is configured. Without a
    key it returns deterministic local signals so demos and tests remain stable.
    """

    api_key: str = ""
    endpoint: str = "https://google.serper.dev/search"
    timeout_s: float = 3.0
    max_results: int = 3

    @classmethod
    def from_env(cls) -> "WebSearchAugmenter":
        return cls(api_key=os.environ.get("SERPER_API_KEY", "").strip())

    def augment(
        self,
        *,
        mission_brief: str,
        scenario_type: str,
        phase_name: str,
        stage_rationale: str,
        action: Dict[str, Any],
        trajectory_archive: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        query = self._build_query(mission_brief, scenario_type, phase_name, action)
        results = self._serper_search(query) if self.api_key else []
        status = "serper" if results else ("offline" if not self.api_key else "empty")
        action_text = self._action_text(action)
        valid_cases = self._valid_use_cases(scenario_type, phase_name, action, stage_rationale, results)
        trajectory_overlaps = self._map_to_trajectories(action_text, trajectory_archive)
        supporting_evidence = [
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "overlap": round(_overlap(action_text, item.get("snippet", "")), 4),
            }
            for item in results
        ]
        return {
            "status": status,
            "query": query,
            "action_description": self._action_description(action, phase_name),
            "valid_use_cases": valid_cases,
            "trajectory_overlaps": trajectory_overlaps,
            "supporting_evidence": supporting_evidence,
            "evidence_count": len(supporting_evidence),
        }

    def _build_query(
        self,
        mission_brief: str,
        scenario_type: str,
        phase_name: str,
        action: Dict[str, Any],
    ) -> str:
        action_type = str(action.get("action_type", "")).strip()
        category = str(action.get("category") or "").strip()
        target = str(action.get("target") or "").strip()
        medication = str(action.get("medication") or "").strip()
        parts = [
            "clinical team coordination",
            scenario_type.replace("_", " "),
            phase_name,
            action_type,
            category,
            target,
            medication,
            mission_brief[:160],
        ]
        return " ".join(part for part in parts if part)

    def _serper_search(self, query: str) -> List[Dict[str, str]]:
        payload = json.dumps({"q": query, "num": self.max_results}).encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=payload,
            headers={
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except (OSError, error.URLError, TimeoutError):
            return []

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        organic = data.get("organic", []) if isinstance(data, dict) else []
        results = []
        for item in organic[: self.max_results]:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": str(item.get("title", "")),
                    "link": str(item.get("link", "")),
                    "snippet": str(item.get("snippet", "")),
                }
            )
        return results

    def _action_text(self, action: Dict[str, Any]) -> str:
        return " ".join(
            str(action.get(key) or "")
            for key in ("action_type", "target", "category", "medication", "message")
        )

    def _action_description(self, action: Dict[str, Any], phase_name: str) -> str:
        action_type = str(action.get("action_type", "") or "move")
        category = str(action.get("category", "") or "coordination")
        target = str(action.get("target", "") or "the council")
        return f"{action_type} move during {phase_name}, aimed at {category} through {target}."

    def _valid_use_cases(
        self,
        scenario_type: str,
        phase_name: str,
        action: Dict[str, Any],
        stage_rationale: str,
        results: Sequence[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        action_text = self._action_text(action)
        base_case = {
            "case": f"{phase_name} in {scenario_type}",
            "signal": stage_rationale,
            "support": round(0.45 + 0.35 * min(1.0, _overlap(action_text, stage_rationale)), 4),
            "source": "environment_stage_graph",
        }
        cases = [base_case]
        for item in results[: self.max_results]:
            snippet = item.get("snippet", "")
            cases.append(
                {
                    "case": item.get("title", "")[:90],
                    "signal": snippet[:220],
                    "support": round(_overlap(action_text, snippet), 4),
                    "source": item.get("link", ""),
                }
            )
        return cases

    def _map_to_trajectories(
        self,
        action_text: str,
        trajectory_archive: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        for item in trajectory_archive:
            summary = str(item.get("summary", ""))
            score = _overlap(action_text, summary)
            if score <= 0.0:
                continue
            ranked.append(
                {
                    "summary": summary,
                    "scenario_type": item.get("scenario_type", ""),
                    "score": round(score, 4),
                }
            )
        ranked.sort(key=lambda row: row["score"], reverse=True)
        return ranked[:3]


def timestamp_ms() -> int:
    return int(time.time() * 1000)
