"""
Cyber Resilience Metrics Tracker
================================
Calculates quantitative security resilience metrics:
- MCI: Mission Criticality Index
- CRI: Cyber Resilience Index
- DREI: Dynamic Redundancy Efficiency Index (incorporating strategic recovery multiplier kappa = 1.2)
"""

from typing import Dict, Any, List, Optional
from src.core.interfaces import PaceState
from config.settings import settings


class MetricsTracker:
    """Evaluates quantitative security resilience metrics grounded in CCRT."""

    @staticmethod
    def calculate_mci(active_threats_count: int, critical_subsystems_affected: int) -> float:
        """Calculates Mission Criticality Index (0.0 to 1.0)."""
        score = (0.3 * active_threats_count) + (0.4 * critical_subsystems_affected)
        return min(1.0, max(0.0, score))

    @staticmethod
    def calculate_cri(recovered_components: int, total_compromised: int) -> float:
        """Calculates Cyber Resilience Index (0.0 to 1.0)."""
        if total_compromised == 0:
            return 1.0
        return min(1.0, float(recovered_components) / float(total_compromised))

    @staticmethod
    def calculate_drei(
        state_utilities: Dict[str, float],
        occupancy_probabilities: Dict[str, float],
        transition_cost: float,
        target_state: PaceState = PaceState.PRIMARY,
    ) -> float:
        """
        Calculates Dynamic Redundancy Efficiency Index (DREI_t) formula:
        DREI_t = Sum_{s in S} (w_t^*(s) * P_t(s)) / C_t
        
        Where w_t^*(s) is scaled by recovery multiplier kappa = 1.2 when returning to PRIMARY state.
        
        Args:
            state_utilities: Utility profile for each PACE state.
            occupancy_probabilities: Probability P_t(s) of being in each PACE state.
            transition_cost: C_t computational and energy transition cost.
            target_state: Target PACE fallback state.
            
        Returns:
            float: Bounded DREI score.
        """
        kappa = settings.STRATEGIC_RECOVERY_MULTIPLIER_KAPPA if target_state == PaceState.PRIMARY else 1.0
        c_t = max(0.01, transition_cost)  # Prevent division by zero

        total_weighted_utility = 0.0
        for state_key, p_val in occupancy_probabilities.items():
            u_val = state_utilities.get(state_key, 0.5)
            if state_key == PaceState.PRIMARY.value or state_key == "PRIMARY":
                u_val *= kappa
            total_weighted_utility += u_val * p_val

        drei_score = total_weighted_utility / c_t
        return round(float(drei_score), 4)
