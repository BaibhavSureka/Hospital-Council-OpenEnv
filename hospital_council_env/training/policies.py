"""Baseline and random policies for Hospital Council evaluation."""

from __future__ import annotations

import random

from hospital_council_env.models import HospitalCouncilAction


PRIMARY_BY_SCENARIO = {
    "diagnostic_ambiguity": "attending_physician",
    "medication_alignment": "pharmacist",
    "conservative_monitoring": "triage_nurse",
    "discharge_negotiation": "bed_manager",
}

SECONDARY_BY_SCENARIO = {
    "diagnostic_ambiguity": "triage_nurse",
    "medication_alignment": "attending_physician",
    "conservative_monitoring": "family_liaison",
    "discharge_negotiation": "family_liaison",
}

ALL_STAKEHOLDERS = [
    "attending_physician",
    "triage_nurse",
    "pharmacist",
    "bed_manager",
    "family_liaison",
]


def baseline_policy(obs) -> HospitalCouncilAction:
    scenario = obs.scenario_type
    phase = obs.phase_name
    meds = list(obs.patient_snapshot.get("candidate_medications", []))
    if phase == "sensemaking":
        return HospitalCouncilAction(
            action_type="consult",
            target=PRIMARY_BY_SCENARIO[scenario],
            message="Give me the most decision-relevant constraint you see right now.",
        )
    if phase == "alignment":
        category = {
            "diagnostic_ambiguity": "diagnosis",
            "medication_alignment": "medication",
            "conservative_monitoring": "no_action",
            "discharge_negotiation": "discharge",
        }[scenario]
        return HospitalCouncilAction(
            action_type="propose",
            target=PRIMARY_BY_SCENARIO[scenario],
            category=category,
            medication=meds[0] if category == "medication" and meds else None,
            message="Here is the current directional plan for the council.",
        )
    if phase == "execution":
        category = {
            "diagnostic_ambiguity": "diagnosis",
            "medication_alignment": "medication",
            "conservative_monitoring": "no_action",
            "discharge_negotiation": "no_action",
        }[scenario]
        return HospitalCouncilAction(
            action_type="commit",
            category=category,
            medication=meds[0] if category == "medication" and meds else None,
            message="Execute the highest-confidence move.",
        )
    if phase == "conflict_resolution":
        return HospitalCouncilAction(
            action_type="resolve",
            target=SECONDARY_BY_SCENARIO[scenario],
            category="discharge" if scenario == "discharge_negotiation" else "no_action",
            message="Reduce friction so the final move lands cleanly.",
        )
    if scenario == "discharge_negotiation":
        return HospitalCouncilAction(
            action_type="commit",
            category="discharge",
            message="Commit to a safe handoff.",
        )
    return HospitalCouncilAction(
        action_type="delegate",
        target=SECONDARY_BY_SCENARIO[scenario],
        category="medication" if scenario == "medication_alignment" else "no_action",
        medication=meds[0] if scenario == "medication_alignment" and meds else None,
        message="Keep the right stakeholder engaged while the plan matures.",
    )


def random_policy(obs, rng: random.Random | None = None) -> HospitalCouncilAction:
    rng = rng or random.Random()
    meds = list(obs.patient_snapshot.get("candidate_medications", []))
    action_type = rng.choice(["consult", "propose", "delegate", "resolve", "commit"])
    category = rng.choice(["diagnosis", "medication", "discharge", "no_action"])
    target = rng.choice(ALL_STAKEHOLDERS)
    return HospitalCouncilAction(
        action_type=action_type,
        target=target if action_type in ("consult", "delegate", "resolve") else None,
        category=category if action_type != "consult" else None,
        medication=meds[0] if category == "medication" and meds else None,
        message=f"Random {action_type} move for {obs.phase_name}.",
        confidence=0.25,
    )
