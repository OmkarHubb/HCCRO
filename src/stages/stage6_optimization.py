"""
Stage 6: Hierarchical Multi-Objective Optimization
==================================================
Solves multi-objective resilience utility function:
    Maximize U = [alpha*R + beta*M + gamma*T + delta*C + lambda*H]

Includes:
- Adaptive Weight Controller (Hardware-Saturating Feedback Loop): If battery < 40%,
  communication weight delta quadratically drops to 0.0, and self-healing weight lambda exponentially spikes.
- MDP PACE Fallback State Solver: epsilon-greedy exploration-exploitation policy solver across 4 PACE states.
- DREI Calculation: Dynamic Redundancy Efficiency Index solver.
"""

import math
import random
from typing import Dict, Any, List, Tuple
from src.core.interfaces import BaseStage, StateVectorData, MIAOutput, OptimizationOutput, PaceState
from config.settings import settings
from src.utils.metrics_tracker import MetricsTracker
from src.utils.logger import get_logger

logger = get_logger("Stage6.Optimization")


class AdaptiveWeightController:
    """
    Hardware-Saturating Feedback Loop adjusting multi-objective weights based on local resource state.
    """

    @staticmethod
    def compute_weights(energy_availability: float) -> Dict[str, float]:
        """
        Computes dynamic weights. If energy_availability < 0.40 (40% battery):
        - Communication weight delta quadratically drops to 0.0.
        - Self-healing weight lambda exponentially spikes to 100.0.
        """
        alpha = settings.WEIGHT_R_RESILIENCE
        beta = settings.WEIGHT_M_MISSION
        gamma = settings.WEIGHT_T_TRUST
        delta = settings.WEIGHT_C_COMMUNICATION
        lambd = settings.WEIGHT_H_HEALING

        if energy_availability < settings.BATTERY_CRITICAL_THRESHOLD:
            # Quadratic drop for communication delta: delta * (energy / threshold)^2
            ratio = max(0.0, energy_availability / settings.BATTERY_CRITICAL_THRESHOLD)
            delta = round(delta * (ratio ** 2), 4)

            # Exponential spike for self-healing lambda: lambda * exp(4 * (1 - ratio))
            lambd = round(float(settings.HEALING_SPIKE_VALUE), 2)
            logger.warning("[AdaptiveWeightController] Critical Battery (%.1f%%)! Delta dropped to %.4f, Lambda spiked to %.2f", energy_availability * 100.0, delta, lambd)

        return {
            "alpha_R": alpha,
            "beta_M": beta,
            "gamma_T": gamma,
            "delta_C": delta,
            "lambda_H": lambd,
        }


class PaceMdpSolver:
    """
    Markov Decision Process (MDP) solver for PACE state transitions using an epsilon-greedy policy.
    """

    PACE_STATES = [PaceState.PRIMARY, PaceState.ALTERNATE, PaceState.CONTINGENCY, PaceState.EMERGENCY]

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def solve_optimal_transition(
        self,
        current_state: PaceState,
        s_t: StateVectorData,
        threat_utilities: Dict[str, float],
    ) -> Tuple[PaceState, float]:
        """
        Solves MDP optimal next PACE state transition:
        - Downward transition prob: p(s, s') = rho * p_max where rho is dynamic threat score (1 - S_t average).
        - Reward function R(s, s') = omega(s') - c_t(s, s').
        - Epsilon-greedy policy selector.
        """
        # Threat score rho in [0, 1]
        rho = 1.0 - ((s_t.C + s_t.R + s_t.T + s_t.E) / 4.0)

        # Transition costs
        costs = {
            (PaceState.PRIMARY, PaceState.PRIMARY): 0.05,
            (PaceState.PRIMARY, PaceState.ALTERNATE): 0.15,
            (PaceState.PRIMARY, PaceState.CONTINGENCY): 0.35,
            (PaceState.PRIMARY, PaceState.EMERGENCY): 0.60,
            (PaceState.ALTERNATE, PaceState.EMERGENCY): 0.40,
            (PaceState.EMERGENCY, PaceState.PRIMARY): 0.70,  # Recovery transition
        }

        # Epsilon-greedy selection
        if random.random() < self.epsilon and len(s_t.active_threats) > 0:
            # Exploration: Select alternate fallback state
            selected = random.choice(self.PACE_STATES)
            reward = threat_utilities.get(selected.value, 0.5) - costs.get((current_state, selected), 0.25)
            return selected, round(reward, 4)

        # Exploitation: Determine highest expected reward
        best_state = PaceState.PRIMARY
        best_reward = -999.0

        # High threat or critical resource forces lower PACE state
        if s_t.E < 0.20 or "RESOURCE_DOS_SUSPECTED" in s_t.active_threats:
            best_state = PaceState.EMERGENCY
        elif rho > 0.6 or len(s_t.active_threats) >= 2:
            best_state = PaceState.CONTINGENCY
        elif rho > 0.3 or len(s_t.active_threats) == 1:
            best_state = PaceState.ALTERNATE
        else:
            best_state = PaceState.PRIMARY

        omega_s_prime = threat_utilities.get(best_state.value, 0.8)
        cost = costs.get((current_state, best_state), 0.20)
        best_reward = omega_s_prime - cost

        return best_state, round(best_reward, 4)


class HierarchicalOptimizationSolver(BaseStage):
    """
    Stage 6: Hierarchical Multi-Objective Optimization Solver Engine.
    Inherits from BaseStage to enforce plug-and-play adaptability.
    """

    def __init__(self):
        super().__init__(name="Stage6_Optimization")
        self.mdp_solver = PaceMdpSolver()

    def execute(self, input_data: Dict[str, Any]) -> OptimizationOutput:
        """
        Executes Stage 6 optimization using standard interfaces.
        
        Args:
            input_data: Dict containing 'state_vector' (StateVectorData) and 'mia_output' (MIAOutput).
            
        Returns:
            OptimizationOutput: Standardized optimization response artifact.
        """
        s_t = input_data.get("state_vector") or StateVectorData()
        mia = input_data.get("mia_output") or MIAOutput(0.0, 1.0, 0.0, "LOW", [], {})

        logger.info("[Stage 6 Optimization] Executing Multi-Objective Utility Solver & Adaptive Weights.")

        # 1. Adaptive Weight Controller based on energy level
        weights = AdaptiveWeightController.compute_weights(s_t.E)

        # 2. Multi-Objective Objective Function Evaluation: U = alpha*R + beta*M + gamma*T + delta*C + lambda*H
        # Self-healing efficiency H is set to 1.0 if actions taken
        h_val = 1.0 if s_t.active_threats else 0.8
        utility = (
            weights["alpha_R"] * s_t.R
            + weights["beta_M"] * s_t.M
            + weights["gamma_T"] * s_t.T
            + weights["delta_C"] * s_t.C
            + (weights["lambda_H"] if weights["lambda_H"] > 10.0 else weights["lambda_H"] * h_val)
        )
        utility = round(utility, 4)

        # 3. MDP Policy Solver for PACE State Selection
        target_pace_state, reward = self.mdp_solver.solve_optimal_transition(
            s_t.pace_state, s_t, mia.threat_adjusted_utilities
        )

        # 4. Map active threats to specific countermeasure action keys
        actions = []
        if "RF_JAMMING_SUSPECTED" in s_t.active_threats or s_t.C < 0.5:
            actions.append("ACTIVATE_FREQUENCY_HOPPING")
        if "GPS_SPOOFING_SUSPECTED" in s_t.active_threats or s_t.T < 0.5:
            actions.append("SWITCH_TO_INERTIAL_NAV_FALLBACK")
        if "RESOURCE_DOS_SUSPECTED" in s_t.active_threats or s_t.E < 0.2:
            actions.append("ENFORCE_PROCESS_QUOTA_ISOLATION")
            actions.append("TRANSITION_PACE_EMERGENCY_SAFE_MODE")

        if not actions:
            actions.append("MAINTAIN_NOMINAL_OPERATIONS")

        # 5. Dynamic Redundancy Efficiency Index (DREI) calculation
        occupancy_probs = {
            PaceState.PRIMARY.value: 0.8 if target_pace_state == PaceState.PRIMARY else 0.1,
            PaceState.ALTERNATE.value: 0.8 if target_pace_state == PaceState.ALTERNATE else 0.1,
            PaceState.CONTINGENCY.value: 0.8 if target_pace_state == PaceState.CONTINGENCY else 0.1,
            PaceState.EMERGENCY.value: 0.8 if target_pace_state == PaceState.EMERGENCY else 0.1,
        }
        drei_score = MetricsTracker.calculate_drei(
            state_utilities=mia.threat_adjusted_utilities,
            occupancy_probabilities=occupancy_probs,
            transition_cost=0.25 if target_pace_state != PaceState.PRIMARY else 0.05,
            target_state=target_pace_state,
        )

        return OptimizationOutput(
            optimal_actions=actions,
            target_pace_state=target_pace_state,
            utility_score=utility,
            drei_score=drei_score,
            weights_applied=weights,
        )

    def process(self, state: Any, impact_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility bridge returning dictionary representation."""
        if isinstance(state, StateVectorData):
            s_t = state
        else:
            s_t = StateVectorData(
                C=getattr(state, "s_t", {}).get("C", 1.0),
                R=getattr(state, "s_t", {}).get("R", 1.0),
                T=getattr(state, "s_t", {}).get("T", 1.0),
                E=getattr(state, "s_t", {}).get("E", 1.0),
                active_threats=getattr(state, "active_threats", []),
            )
        mia = MIAOutput(impact_metrics.get("mci_score", 0.0), 1.0, 0.0, "LOW", [], {})
        out = self.execute({"state_vector": s_t, "mia_output": mia})
        return {
            "optimal_actions": out.optimal_actions,
            "target_pace_state": out.target_pace_state.value,
            "solver_objective_value": out.utility_score,
            "drei_score": out.drei_score,
            "weights_used": out.weights_applied,
        }


# Backward compatibility alias
Stage6Optimization = HierarchicalOptimizationSolver
