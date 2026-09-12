"""
HCCRO Pipeline Skeleton — 8-Stage Cognitive Architecture Pipeline Execution Core
"""

from src.core.orchestrator import HCCROOrchestrator
from src.data.opsat_parser import RealWorldOPSSATParser
from src.engine.unknown_attack_engine import UnknownAttackAdaptiveEngine

__all__ = [
    "HCCROOrchestrator",
    "RealWorldOPSSATParser",
    "UnknownAttackAdaptiveEngine",
]


def run_pipeline_with_zero_day_fallback(
    orchestrator: HCCROOrchestrator,
    raw_telemetry: dict,
    enable_zero_day_engine: bool = False
) -> dict:
    """
    Non-breaking pipeline execution hook with zero-day fallback.
    If enable_zero_day_engine=False (default), existing execution paths remain 100% untouched.
    """
    result = orchestrator.execute_pipeline(raw_telemetry)

    if not enable_zero_day_engine:
        return result

    engine = UnknownAttackAdaptiveEngine()
    state_vector = result["state_vector"]
    aim_out = result["aim_output"]
    prob_dist = getattr(aim_out, "attack_type_probabilities", {})

    if engine.detect_open_set_threat(prob_dist, state_vector):
        fingerprint = engine.build_fingerprint(state_vector, raw_telemetry)
        candidates = engine.search_cke_memory(fingerprint)
        optimal_act = engine.evaluate_counterfactuals(state_vector, candidates)
        result["mitigation_plan"].optimal_actions = [optimal_act]
        engine.persist_experience(fingerprint, optimal_act, mci_gain=0.15, cri_gain=0.20, success=True)

    return result
