"""
Stage 5: Mission Impact Estimation (MIA)
=========================================
Quantifies operational risk, calculates Mission Degradation Index (MDI), MCI, CRI,
and computes threat-adjusted state utilities omega_t*(s) for Stage 6 solver.
"""

from typing import Dict, Any, List
from src.core.interfaces import BaseStage, StateVectorData, AEPOutput, MIAOutput, PaceState
from src.utils.metrics_tracker import MetricsTracker
from src.utils.logger import get_logger

logger = get_logger("Stage5.MIA")


class MissionImpactEstimation(BaseStage):
    """
    Stage 5: Mission Impact Estimation (MIA) Engine.
    Inherits from BaseStage to enforce plug-and-play adaptability.
    """

    def __init__(self):
        super().__init__(name="Stage5_MIA")

    def _calculate_impact_metrics(
        self, s_t: StateVectorData, aep: AEPOutput
    ) -> MIAOutput:
        """
        Calculates MDI, MCI, CRI, and threat-adjusted PACE state utilities omega_t*(s).
        """
        num_threats = len(s_t.active_threats)
        num_paths = len(aep.predicted_paths) if aep.predicted_paths else 0

        # Mission Degradation Index (MDI)
        mdi_score = round(min(1.0, 0.25 * num_threats + 0.15 * aep.escalation_risk), 3)

        # MCI and CRI via MetricsTracker
        mci_score = MetricsTracker.calculate_mci(num_threats, num_paths)
        cri_score = MetricsTracker.calculate_cri(
            recovered_components=0 if num_threats > 0 else 1,
            total_compromised=max(1, num_threats),
        )

        affected_subsystems = [
            p.get("target", "Payload_Controller") for p in aep.predicted_paths if "target" in p
        ]

        # Threat-adjusted utility values omega_t*(s) for PACE states
        # Baseline utilities: PRIMARY=1.0, ALTERNATE=0.75, CONTINGENCY=0.45, EMERGENCY=0.15
        threat_penalty = mdi_score * 0.40
        threat_adjusted_utilities = {
            PaceState.PRIMARY.value: round(max(0.1, 1.0 - threat_penalty), 3),
            PaceState.ALTERNATE.value: round(max(0.1, 0.75 - threat_penalty * 0.5), 3),
            PaceState.CONTINGENCY.value: round(max(0.1, 0.45), 3),
            PaceState.EMERGENCY.value: round(max(0.05, 0.15), 3),
        }

        severity = "CRITICAL" if mdi_score > 0.6 else ("HIGH" if mdi_score > 0.3 else "LOW")

        return MIAOutput(
            mci_score=mci_score,
            cri_score=cri_score,
            mdi_score=mdi_score,
            threat_severity=severity,
            affected_subsystems=list(set(affected_subsystems)),
            threat_adjusted_utilities=threat_adjusted_utilities,
        )

    def execute(self, input_data: Dict[str, Any]) -> MIAOutput:
        """
        Executes Stage 5 impact estimation using standard interfaces.
        
        Args:
            input_data: Dict containing 'state_vector' (StateVectorData) and 'aep_output' (AEPOutput).
            
        Returns:
            MIAOutput: Standardized impact estimation artifact.
        """
        s_t = input_data.get("state_vector") or StateVectorData()
        aep = input_data.get("aep_output") or AEPOutput([], {}, 0.0)

        logger.info("[Stage 5 MIA] Calculating Mission Impact Estimation (MDI) & threat utilities.")
        return self._calculate_impact_metrics(s_t, aep)

    def process(self, state: Any, evolution_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Backward compatibility bridge returning dictionary representation."""
        if isinstance(state, StateVectorData):
            s_t = state
        else:
            s_t = StateVectorData(active_threats=getattr(state, "active_threats", []))
        aep = AEPOutput(evolution_paths, {}, 0.5 if evolution_paths else 0.0)
        output = self._calculate_impact_metrics(s_t, aep)
        return {
            "mci_score": output.mci_score,
            "threat_severity": output.threat_severity,
            "impacted_subsystems": output.affected_subsystems,
        }


# Backward compatibility alias
Stage5MIA = MissionImpactEstimation
