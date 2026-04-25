"""Self-contained LLM-style search augmentation and context synthesis."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


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


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def timestamp_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class LLMSearchAugmenter:
    """Simulates retrieval by generating ranked pseudo-documents from prompts."""

    max_results: int = 4

    @classmethod
    def from_env(cls) -> "LLMSearchAugmenter":
        return cls()

    def augment(
        self,
        *,
        mission_brief: str,
        scenario_type: str,
        phase_name: str,
        stage_rationale: str,
        action: Dict[str, Any],
        patient_snapshot: Dict[str, Any],
        expected_action: Dict[str, Any],
        trajectory_archive: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        query = self._build_query(mission_brief, scenario_type, phase_name, action, patient_snapshot)
        action_text = self._action_text(action)
        search_prompt = self._build_search_prompt(query, action, expected_action, patient_snapshot)
        candidates = self._generate_candidates(
            query=query,
            phase_name=phase_name,
            stage_rationale=stage_rationale,
            action=action,
            patient_snapshot=patient_snapshot,
            expected_action=expected_action,
        )
        ranked_results = self._rank_candidates(query, action_text, expected_action, candidates)
        trajectory_alignments = self._map_to_trajectories(query, action_text, trajectory_archive)
        consistency_checks = self._consistency_checks(action, expected_action, ranked_results, trajectory_alignments)
        valid_use_cases = [
            {
                "case": item["title"],
                "signal": item["applicability"],
                "support": item["relevance"],
                "source": "llm_pseudo_search",
            }
            for item in ranked_results
        ]
        supporting_evidence = [
            {
                "title": item["title"],
                "snippet": item["summary"],
                "definition": item["definition"],
                "usage_contexts": item["usage_contexts"],
                "constraints": item["constraints"],
                "applicability": item["applicability"],
                "valid_scenarios": item["valid_scenarios"],
                "invalid_scenarios": item["invalid_scenarios"],
                "relevance": item["relevance"],
            }
            for item in ranked_results
        ]
        return {
            "status": "llm_simulated_search",
            "query": query,
            "search_prompt": search_prompt,
            "ranking_prompt": self._build_ranking_prompt(query, ranked_results),
            "trajectory_prompt": self._build_trajectory_prompt(query, action, trajectory_archive),
            "action_description": self._action_description(action, phase_name),
            "pseudo_results": candidates,
            "ranked_results": ranked_results,
            "trajectory_alignments": trajectory_alignments,
            "trajectory_overlaps": trajectory_alignments,
            "consistency_checks": consistency_checks,
            "valid_use_cases": valid_use_cases,
            "supporting_evidence": supporting_evidence,
            "evidence_count": len(ranked_results),
        }

    def _build_query(
        self,
        mission_brief: str,
        scenario_type: str,
        phase_name: str,
        action: Dict[str, Any],
        patient_snapshot: Dict[str, Any],
    ) -> str:
        parts = [
            "clinical coordination search",
            scenario_type.replace("_", " "),
            phase_name,
            str(action.get("action_type", "") or ""),
            str(action.get("category", "") or ""),
            str(action.get("target", "") or ""),
            " ".join(patient_snapshot.get("salient_labs", [])[:3]),
            mission_brief[:160],
        ]
        return " ".join(part for part in parts if str(part).strip())

    def _build_search_prompt(
        self,
        query: str,
        action: Dict[str, Any],
        expected_action: Dict[str, Any],
        patient_snapshot: Dict[str, Any],
    ) -> str:
        return (
            "You are simulating a search engine for hospital coordination. "
            "Generate multiple candidate results for the query-action pair. "
            "Each result must include a definition, usage contexts, constraints, "
            "applicability, and scenarios where the action is valid or invalid. "
            f"Query: {query}. "
            f"Action: {action}. Expected: {expected_action}. "
            f"Patient snapshot: {patient_snapshot}."
        )

    def _build_ranking_prompt(self, query: str, ranked_results: Sequence[Dict[str, Any]]) -> str:
        return (
            "Rank pseudo-search results by relevance to the current query and action. "
            f"Query: {query}. Candidates: {len(ranked_results)}."
        )

    def _build_trajectory_prompt(
        self,
        query: str,
        action: Dict[str, Any],
        trajectory_archive: Sequence[Dict[str, Any]],
    ) -> str:
        return (
            "Identify semantic overlaps between the current query-action pair and prior trajectories. "
            f"Query: {query}. Action: {action}. Archive size: {len(trajectory_archive)}."
        )

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

    def _generate_candidates(
        self,
        *,
        query: str,
        phase_name: str,
        stage_rationale: str,
        action: Dict[str, Any],
        patient_snapshot: Dict[str, Any],
        expected_action: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        action_type = str(action.get("action_type", "") or "move")
        category = str(action.get("category", "") or "no_action")
        target = str(action.get("target", "") or "council")
        labs = list(patient_snapshot.get("salient_labs", [])[:3])
        lab_phrase = ", ".join(labs) if labs else "no dominant lab cue"
        category_phrase = category.replace("_", " ")
        expected_category = str(expected_action.get("category", "") or category_phrase).replace("_", " ")
        candidates = [
            {
                "title": f"{phase_name.title()} {action_type} for {category_phrase}",
                "definition": f"{action_type.title()} is used to move the council through the {phase_name} phase.",
                "usage_contexts": [
                    f"When the coordinator needs progress on {category_phrase}.",
                    f"When {target} has decision-relevant information.",
                ],
                "constraints": [
                    "Should fit the active phase rationale.",
                    "Should not break medication or discharge safety gates.",
                ],
                "applicability": stage_rationale,
                "valid_scenarios": [f"{phase_name} in {category_phrase}", f"Lab cues: {lab_phrase}"],
                "invalid_scenarios": ["Premature irreversible action", "Ignoring active coalition friction"],
                "summary": f"{action_type} helps when {stage_rationale.lower()}",
            },
            {
                "title": f"Expected path for {expected_category}",
                "definition": f"The strongest next move usually preserves alignment with {expected_category}.",
                "usage_contexts": [
                    f"When the expected action is {expected_action.get('action_type', action_type)}.",
                    "When historical cases reward stage-consistent actions.",
                ],
                "constraints": [
                    "Requires semantic fit with the ground-truth direction.",
                    "Mismatch raises correction pressure.",
                ],
                "applicability": f"Most relevant when the council should steer toward {expected_category}.",
                "valid_scenarios": [f"Expected {expected_category}", f"Query: {query[:90]}"],
                "invalid_scenarios": ["Category drift", "Targeting the wrong stakeholder repeatedly"],
                "summary": f"Expected action alignment is strongest for {expected_category}.",
            },
            {
                "title": f"Stakeholder interpretation for {target}",
                "definition": f"{target.replace('_', ' ').title()} interprets the move through operational and safety constraints.",
                "usage_contexts": [
                    "When coalition support matters for the next step.",
                    "When one stakeholder is the main source of uncertainty reduction.",
                ],
                "constraints": [
                    "Low support if the target is missing or misaligned.",
                    "Repeated weak targeting reduces efficiency.",
                ],
                "applicability": f"Useful if {target} is the right stakeholder for this phase.",
                "valid_scenarios": [f"Consult or resolve with {target}", "High-friction steps"],
                "invalid_scenarios": ["Target omitted during stakeholder-specific phases"],
                "summary": f"Stakeholder choice changes the quality of {action_type}.",
            },
            {
                "title": f"Lab-aware interpretation for {category_phrase}",
                "definition": "Clinical coordination should reflect whether lab evidence points toward escalation, monitoring, or discharge.",
                "usage_contexts": [
                    f"When salient labs are {lab_phrase}.",
                    "When abnormal lab burden adds diagnostic uncertainty.",
                ],
                "constraints": [
                    "Strong abnormal labs favor sensemaking or diagnosis before discharge.",
                    "Medication without evidence should be penalized.",
                ],
                "applicability": f"Applicable when lab burden supports or weakens {category_phrase}.",
                "valid_scenarios": [f"Lab cues: {lab_phrase}", "Abnormal lab count above baseline"],
                "invalid_scenarios": ["Discharge under unresolved abnormal labs"],
                "summary": f"Lab cues suggest whether {category_phrase} is justified now.",
            },
        ]
        return candidates[: self.max_results]

    def _rank_candidates(
        self,
        query: str,
        action_text: str,
        expected_action: Dict[str, Any],
        candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        expected_text = " ".join(str(expected_action.get(key) or "") for key in ("action_type", "category", "target"))
        ranked = []
        for index, item in enumerate(candidates):
            reference_text = " ".join(
                [item["title"], item["definition"], item["applicability"], item["summary"]]
                + list(item.get("usage_contexts", []))
                + list(item.get("constraints", []))
            )
            relevance = _bounded(
                0.45 * _overlap(query, reference_text)
                + 0.35 * _overlap(action_text, reference_text)
                + 0.20 * _overlap(expected_text, reference_text)
            )
            ranked_item = dict(item)
            ranked_item["rank"] = index + 1
            ranked_item["relevance"] = round(relevance, 4)
            ranked.append(ranked_item)
        ranked.sort(key=lambda row: row["relevance"], reverse=True)
        for index, item in enumerate(ranked):
            item["rank"] = index + 1
        return ranked[: self.max_results]

    def _map_to_trajectories(
        self,
        query: str,
        action_text: str,
        trajectory_archive: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        for item in trajectory_archive:
            summary = str(item.get("summary", ""))
            overlap_score = _bounded(0.55 * _overlap(query, summary) + 0.45 * _overlap(action_text, summary))
            if overlap_score <= 0.0:
                continue
            ranked.append(
                {
                    "summary": summary,
                    "scenario_type": str(item.get("scenario_type", "")),
                    "semantic_overlap": round(overlap_score, 4),
                    "task_graph_loss": float(item.get("task_graph_loss", 0.0)),
                }
            )
        ranked.sort(key=lambda row: row["semantic_overlap"], reverse=True)
        return ranked[:3]

    def _consistency_checks(
        self,
        action: Dict[str, Any],
        expected_action: Dict[str, Any],
        ranked_results: Sequence[Dict[str, Any]],
        trajectory_alignments: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        action_type_match = str(action.get("action_type", "")) == str(expected_action.get("action_type", ""))
        category_match = str(action.get("category", "")) == str(expected_action.get("category", ""))
        support = _mean([float(item.get("relevance", 0.0)) for item in ranked_results[:2]])
        overlap = _mean([float(item.get("semantic_overlap", 0.0)) for item in trajectory_alignments[:2]])
        return {
            "action_type_match": action_type_match,
            "category_match": category_match,
            "search_support": round(support, 4),
            "trajectory_support": round(overlap, 4),
            "overall_consistency": round(_bounded(0.5 * support + 0.3 * overlap + 0.2 * float(action_type_match and category_match)), 4),
        }


WebSearchAugmenter = LLMSearchAugmenter


@dataclass
class ContextLLMManager:
    """Synthesizes evaluation context into an actionable next-step observation."""

    max_trajectories: int = 3

    def build_observation(
        self,
        *,
        action: Dict[str, Any],
        expected_action: Dict[str, Any],
        next_expected_action: Dict[str, Any],
        retrieved_trajectories: Sequence[Dict[str, Any]],
        llm_search: Dict[str, Any],
        task_graph: Dict[str, Any],
        task_graph_loss: float,
        milestone_score: float,
        safety_score: float,
        coalition_score: float,
        patient_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized = self._normalize_inputs(action, expected_action, next_expected_action, patient_snapshot)
        high_signal_trajectories = self._rank_trajectories(normalized, retrieved_trajectories)
        synthesis = self._semantic_synthesis(normalized, high_signal_trajectories, llm_search)
        classification_payload = self._classify(
            normalized=normalized,
            llm_search=llm_search,
            task_graph_loss=task_graph_loss,
            milestone_score=milestone_score,
            safety_score=safety_score,
            coalition_score=coalition_score,
            high_signal_trajectories=high_signal_trajectories,
        )
        correction_signal = self._correction_signal(normalized, classification_payload["classification"])
        guidance = self._next_step_guidance(classification_payload["classification"], classification_payload["confidence"])
        return {
            "manager_status": "context_llm_manager",
            "normalized_inputs": normalized,
            "high_signal_trajectories": high_signal_trajectories,
            "semantic_synthesis": synthesis,
            "classification": classification_payload["classification"],
            "class_probabilities": classification_payload["class_probabilities"],
            "confidence": classification_payload["confidence"],
            "diagnostic_explanation": classification_payload["diagnostic_explanation"],
            "correction_signal": correction_signal,
            "next_step_guidance": guidance,
            "recommended_next_action": normalized["next_expected_action"],
            "decision_trace": {
                "task_graph": {
                    "active_node": task_graph.get("active_node", ""),
                    "loss": round(task_graph_loss, 4),
                },
                "search_consistency": llm_search.get("consistency_checks", {}),
            },
            "observation_text": (
                f"{classification_payload['classification']} with confidence "
                f"{classification_payload['confidence']:.3f}. "
                f"Guidance: {guidance}."
            ),
        }

    def _normalize_inputs(
        self,
        action: Dict[str, Any],
        expected_action: Dict[str, Any],
        next_expected_action: Dict[str, Any],
        patient_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "action": {
                "action_type": str(action.get("action_type", "") or ""),
                "category": str(action.get("category", "") or ""),
                "target": str(action.get("target", "") or ""),
                "medication": str(action.get("medication", "") or ""),
                "message": str(action.get("message", "") or ""),
            },
            "expected_action": {
                "action_type": str(expected_action.get("action_type", "") or ""),
                "category": str(expected_action.get("category", "") or ""),
                "target": str(expected_action.get("target", "") or ""),
                "medication": str(expected_action.get("medication", "") or ""),
                "message": str(expected_action.get("message", "") or ""),
            },
            "next_expected_action": {
                "action_type": str(next_expected_action.get("action_type", "") or ""),
                "category": str(next_expected_action.get("category", "") or ""),
                "target": str(next_expected_action.get("target", "") or ""),
                "medication": str(next_expected_action.get("medication", "") or ""),
                "message": str(next_expected_action.get("message", "") or ""),
            },
            "patient_snapshot": {
                "salient_labs": list(patient_snapshot.get("salient_labs", [])[:3]),
                "abnormal_lab_signal_count": int(patient_snapshot.get("abnormal_lab_signal_count", 0)),
                "candidate_medications": list(patient_snapshot.get("candidate_medications", [])[:3]),
            },
        }

    def _rank_trajectories(
        self,
        normalized: Dict[str, Any],
        retrieved_trajectories: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        query_text = " ".join(
            [
                normalized["action"]["action_type"],
                normalized["action"]["category"],
                normalized["expected_action"]["action_type"],
                normalized["expected_action"]["category"],
            ]
        )
        ranked = []
        for item in retrieved_trajectories:
            summary = str(item.get("summary", item.get("case", "")))
            score = _bounded(
                0.6 * float(item.get("semantic_overlap", item.get("score", 0.0)))
                + 0.4 * _overlap(query_text, summary)
            )
            ranked.append(
                {
                    "summary": summary,
                    "scenario_type": str(item.get("scenario_type", "")),
                    "score": round(score, 4),
                }
            )
        ranked.sort(key=lambda row: row["score"], reverse=True)
        return ranked[: self.max_trajectories]

    def _semantic_synthesis(
        self,
        normalized: Dict[str, Any],
        high_signal_trajectories: Sequence[Dict[str, Any]],
        llm_search: Dict[str, Any],
    ) -> Dict[str, Any]:
        action = normalized["action"]
        expected = normalized["expected_action"]
        return {
            "attempted": f"{action['action_type']} toward {action['category'] or 'unspecified'} via {action['target'] or 'no target'}",
            "expected": f"{expected['action_type']} toward {expected['category'] or 'unspecified'} via {expected['target'] or 'no target'}",
            "similar_cases": [item["summary"] for item in high_signal_trajectories[:2]],
            "uncertainty": llm_search.get("consistency_checks", {}),
            "mismatch": {
                "action_type": action["action_type"] != expected["action_type"],
                "category": action["category"] != expected["category"],
                "target": bool(expected["target"]) and action["target"] != expected["target"],
            },
        }

    def _classify(
        self,
        *,
        normalized: Dict[str, Any],
        llm_search: Dict[str, Any],
        task_graph_loss: float,
        milestone_score: float,
        safety_score: float,
        coalition_score: float,
        high_signal_trajectories: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        action = normalized["action"]
        expected = normalized["expected_action"]
        action_type_match = float(action["action_type"] == expected["action_type"])
        category_match = float(action["category"] == expected["category"])
        target_match = float(not expected["target"] or action["target"] == expected["target"])
        search_support = float(llm_search.get("consistency_checks", {}).get("search_support", 0.0))
        trajectory_support = _mean([float(item.get("score", 0.0)) for item in high_signal_trajectories[:2]])
        semantic_score = _bounded(
            0.25 * action_type_match
            + 0.20 * category_match
            + 0.10 * target_match
            + 0.15 * search_support
            + 0.10 * trajectory_support
            + 0.10 * milestone_score
            + 0.05 * safety_score
            + 0.05 * coalition_score
            - 0.20 * task_graph_loss
        )
        correct = _bounded(semantic_score)
        partial = _bounded(1.0 - abs(semantic_score - 0.5) * 1.8)
        incorrect = _bounded(1.0 - semantic_score + 0.25 * task_graph_loss)
        total = correct + partial + incorrect
        probs = {
            "correct": correct / total,
            "partially_correct": partial / total,
            "incorrect": incorrect / total,
        }
        classification = max(probs, key=probs.get)
        confidence = float(probs[classification])
        explanation = (
            "The action was evaluated against expected phase behavior, lab-aware search support, "
            f"trajectory overlap, and task-graph loss. Semantic score={semantic_score:.3f}, "
            f"search_support={search_support:.3f}, trajectory_support={trajectory_support:.3f}."
        )
        return {
            "classification": classification,
            "class_probabilities": {key: round(value, 4) for key, value in probs.items()},
            "confidence": round(confidence, 4),
            "diagnostic_explanation": explanation,
        }

    def _correction_signal(
        self,
        normalized: Dict[str, Any],
        classification: str,
    ) -> Dict[str, Any]:
        expected = normalized["expected_action"]
        next_expected = normalized["next_expected_action"]
        if classification == "correct":
            signal = next_expected
            rationale = "Retain the current strategy and roll into the next expected phase."
        else:
            signal = expected
            rationale = "Correct toward the expected action before further drift compounds."
        return {
            "suggested_action": signal,
            "rationale": rationale,
        }

    def _next_step_guidance(self, classification: str, confidence: float) -> str:
        if classification == "correct" and confidence >= 0.45:
            return "retain"
        if classification == "correct":
            return "refine"
        if classification == "partially_correct":
            return "refine"
        return "replace"
