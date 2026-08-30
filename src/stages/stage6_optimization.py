"""
Stage 6: Hierarchical Multi-Objective Optimization (The Decision Brain)
====================================================================
Migrates and unifies mathematical solvers into Stage6ResilienceOptimizer.

Solves multi-objective resilience utility function:
    Maximize U = [alpha*R + beta*M + gamma*T + delta*C + lambda*H]

Includes:
1. DCS-MOS Dynamic Weight Controller: Auto-adjusts and normalizes weights based on physical battery
   and CPU constraints.
2. Epsilon-Greedy PACE MDP Solver: Evaluates PACE state transitions with dynamic transition costs,
   strategic recovery multiplier (kappa = 1.2), and state-dependent zero exploration (epsilon = 0.0) in Emergency.
3. Assistance Feasibility Score (AFS) Bidding: Peer-to-peer workload offloading evaluation when local CPU load > 70%.
"""

import math
import random
from typing import Dict, Any, List, Tuple, Optional
from src.core.interfaces import BaseStage, StateVectorData, MIAOutput, OptimizationOutput, PaceState
from config.settings import settings
from src.utils.metrics_tracker import MetricsTracker
from src.utils.logger import get_logger

logger = get_logger("Stage6.Optimization")


class AdaptiveWeightController:
    """
    DCS-MOS Dynamic Constraint-Saturating Multi-Objective Scaling Weight Controller.
    Auto-adjusts weights based on physical energy and CPU constraints, normalizing weights.
    """

    @staticmethod
    def compute_weights(
        energy_availability: float, cpu_load: float = 0.20
    ) -> Dict[str, float]:
        """
        Computes dynamic weights:
        - If energy_availability < settings.BATTERY_CRITICAL_THRESHOLD (0.40):
          - Communication weight delta quadratically drops to 0.0: delta * (E / threshold)^2
          - Self-healing weight lambda exponentially spikes to 100.0.
        - If cpu_load > 0.70 (70% CPU usage / DoS):
          - Mission weight beta quadratically drops.
          - Self-healing weight lambda spikes up.
        """
        alpha = float(settings.WEIGHT_R_RESILIENCE)
        beta = float(settings.WEIGHT_M_MISSION)
        gamma = float(settings.WEIGHT_T_TRUST)
        delta = float(settings.WEIGHT_C_COMMUNICATION)
        lambd = float(settings.WEIGHT_H_HEALING)

        crit_threshold = getattr(settings, "BATTERY_CRITICAL_THRESHOLD", 0.40)
        
        if energy_availability < crit_threshold:
            ratio = max(0.0, energy_availability / crit_threshold)
            delta = round(delta * (ratio ** 2), 4)
            lambd = round(float(getattr(settings, "HEALING_SPIKE_VALUE", 100.0)), 2)
            logger.warning(
                "[AdaptiveWeightController] Critical Battery (%.1f%%)! Delta dropped to %.4f, Lambda spiked to %.2f",
                energy_availability * 100.0, delta, lambd
            )

        # High CPU / DoS Constraint
        if cpu_load > 0.70:
            cpu_margin = max(0.0, 1.0 - cpu_load)
            beta = round(beta * ((cpu_margin / 0.30) ** 2), 4)
            if lambd < 50.0:
                lambd = 50.0
            logger.warning(
                "[AdaptiveWeightController] CPU Load Spike (%.1f%%)! Beta dropped to %.4f",
                cpu_load * 100.0, beta
            )

        return {
            "alpha_R": alpha,
            "beta_M": beta,
            "gamma_T": gamma,
            "delta_C": delta,
            "lambda_H": lambd,
        }

    @staticmethod
    def get_normalized_weights(raw_weights: Dict[str, float]) -> Dict[str, float]:
        """Normalizes weights so that alpha + beta + gamma + delta + lambda = 1.0."""
        total = sum(raw_weights.values())
        if total <= 0.0:
            return {k: 0.2 for k in raw_weights}
        return {k: round(v / total, 4) for k, v in raw_weights.items()}


class PaceMdpSolver:
    """
    Markov Decision Process (MDP) solver for PACE state transitions using an epsilon-greedy policy
    with state-dependent epsilon safety constraints and dynamic transition cost modeling.
    """

    PACE_STATES = [PaceState.PRIMARY, PaceState.ALTERNATE, PaceState.CONTINGENCY, PaceState.EMERGENCY]
    PACE_HIERARCHY_LEVELS = {
        PaceState.EMERGENCY: 0,
        PaceState.CONTINGENCY: 1,
        PaceState.ALTERNATE: 2,
        PaceState.PRIMARY: 3,
    }

    def __init__(self, epsilon: float = 0.1):
        self.base_epsilon = epsilon

    def get_transition_cost(
        self, current_state: PaceState, target_state: PaceState, active_threats: List[str]
    ) -> float:
        """Computes transition cost c(s, s') with environmental/threat multipliers."""
        base_costs = {
            (PaceState.PRIMARY, PaceState.PRIMARY): 0.05,
            (PaceState.PRIMARY, PaceState.ALTERNATE): 0.15,
            (PaceState.PRIMARY, PaceState.CONTINGENCY): 0.35,
            (PaceState.PRIMARY, PaceState.EMERGENCY): 0.60,
            (PaceState.ALTERNATE, PaceState.PRIMARY): 0.20,
            (PaceState.ALTERNATE, PaceState.ALTERNATE): 0.05,
            (PaceState.ALTERNATE, PaceState.CONTINGENCY): 0.25,
            (PaceState.ALTERNATE, PaceState.EMERGENCY): 0.40,
            (PaceState.CONTINGENCY, PaceState.CONTINGENCY): 0.05,
            (PaceState.CONTINGENCY, PaceState.EMERGENCY): 0.30,
            (PaceState.EMERGENCY, PaceState.EMERGENCY): 0.05,
            (PaceState.EMERGENCY, PaceState.PRIMARY): 0.70,
        }
        cost = base_costs.get((current_state, target_state), 0.25)
        
        # Environmental adjustment: RF Jamming increases comm/transition costs
        if "RF_JAMMING_SUSPECTED" in active_threats or "RF_JAMMING_DISRUPTION" in active_threats:
            cost *= 1.5

        return round(cost, 4)

    def solve_optimal_transition(
        self,
        current_state: PaceState,
        s_t: StateVectorData,
        threat_utilities: Dict[str, float],
    ) -> Tuple[PaceState, float]:
        """
        Solves MDP optimal next PACE state transition:
        - State-dependent safety policy: Reduce epsilon to 0.0 in Emergency mode.
        - Strategic recovery multiplier kappa = 1.2 applied to upward recovery transitions.
        """
        # Enforce state-dependent epsilon safety policy
        if current_state == PaceState.EMERGENCY or s_t.E < 0.20:
            effective_epsilon = 0.0  # Zero exploration risk to guarantee physical survival
        else:
            effective_epsilon = self.base_epsilon

        # Threat score rho
        rho = 1.0 - ((s_t.C + s_t.R + s_t.T + s_t.E) / 4.0)

        # Exploration choice
        if effective_epsilon > 0.0 and random.random() < effective_epsilon and len(s_t.active_threats) > 0:
            selected = random.choice(self.PACE_STATES)
            cost = self.get_transition_cost(current_state, selected, s_t.active_threats)
            reward = threat_utilities.get(selected.value, 0.5) - cost
            return selected, round(reward, 4)

        # Deterministic exploitation choice based on threat & resource state
        if s_t.E < 0.20 or "RESOURCE_DOS_SUSPECTED" in s_t.active_threats:
            target_state = PaceState.EMERGENCY
        elif rho > 0.6 or len(s_t.active_threats) >= 2:
            target_state = PaceState.CONTINGENCY
        elif rho > 0.3 or len(s_t.active_threats) == 1:
            target_state = PaceState.ALTERNATE
        else:
            target_state = PaceState.PRIMARY

        omega = threat_utilities.get(target_state.value, 0.8)
        cost = self.get_transition_cost(current_state, target_state, s_t.active_threats)

        # Strategic recovery multiplier kappa = 1.2 when transitioning up hierarchy
        curr_lvl = self.PACE_HIERARCHY_LEVELS.get(current_state, 0)
        targ_lvl = self.PACE_HIERARCHY_LEVELS.get(target_state, 0)
        if targ_lvl > curr_lvl:
            kappa = getattr(settings, "STRATEGIC_RECOVERY_MULTIPLIER_KAPPA", 1.2)
            reward = (omega * kappa) - cost
        else:
            reward = omega - cost

        return target_state, round(reward, 4)


class AFSBiddingEngine:
    """
    Assistance Feasibility Score (AFS) Workload Bidding Engine for peer-to-peer offloading.
    """

    @staticmethod
    def evaluate_afs_bidding(
        neighbors: Optional[List[Dict[str, float]]] = None, local_cpu_load: float = 0.80
    ) -> Tuple[Optional[str], float]:
        """
        Evaluates peer satellite bids if local CPU load > 70%.
        AFS = 0.4 * battery_margin + 0.4 * cpu_margin + 0.2 * trust_score.
        Returns (best_neighbor_id, highest_afs_score).
        """
        if local_cpu_load <= 0.70:
            return None, 0.0

        if not neighbors:
            # Simulated cluster neighbor bids
            neighbors = [
                {"id": "SAT_PEER_01", "battery_margin": 0.85, "cpu_margin": 0.75, "trust_score": 0.95},
                {"id": "SAT_PEER_02", "battery_margin": 0.60, "cpu_margin": 0.40, "trust_score": 0.80},
                {"id": "SAT_PEER_03", "battery_margin": 0.90, "cpu_margin": 0.80, "trust_score": 0.90},
            ]

        best_peer = None
        best_score = -1.0

        for peer in neighbors:
            afs = (
                0.4 * peer.get("battery_margin", 0.5)
                + 0.4 * peer.get("cpu_margin", 0.5)
                + 0.2 * peer.get("trust_score", 0.5)
            )
            if afs > best_score:
                best_score = afs
                best_peer = peer.get("id", "SAT_PEER_UNKNOWN")

        return best_peer, round(best_score, 4)


class Stage6ResilienceOptimizer(BaseStage):
    """
    Stage 6: Unified Hierarchical Multi-Objective Resilience Optimizer (The Decision Brain).
    Inherits from BaseStage to enforce plug-and-play adaptability.
    """

    def __init__(self):
        super().__init__(name="Stage6_Optimization")
        self.mdp_solver = PaceMdpSolver()
        self.afs_engine = AFSBiddingEngine()

    def execute(self, input_data: Dict[str, Any]) -> OptimizationOutput:
        """
        Executes Stage 6 optimization using standardized interfaces.
        
        Args:
            input_data: Dict containing 'state_vector' (StateVectorData) and optional 'mia_output' (MIAOutput).
            
        Returns:
            OptimizationOutput: Standardized optimization response artifact.
        """
        s_t = input_data.get("state_vector") or StateVectorData()
        mia = input_data.get("mia_output") or MIAOutput(0.0, 1.0, 0.0, "LOW", [], {})

        logger.info("[Stage 6 Optimization] Ingesting State Vector S_t={C:%.2f, R:%.2f, T:%.2f, Q:%.2f, M:%.2f, E:%.2f, A:%.2f}",
                    s_t.C, s_t.R, s_t.T, s_t.Q, s_t.M, s_t.E, s_t.A)

        # 1. Ingest CPU load and energy to compute DCS-MOS dynamic weights
        cpu_load = max(0.0, min(1.0, 1.0 - s_t.E))
        if "RESOURCE_DOS_SUSPECTED" in s_t.active_threats:
            cpu_load = max(cpu_load, 0.85)

        raw_weights = AdaptiveWeightController.compute_weights(s_t.E, cpu_load=cpu_load)
        norm_weights = AdaptiveWeightController.get_normalized_weights(raw_weights)

        # 2. Compute Multi-Objective Utility Function score U
        h_val = 1.0 if s_t.active_threats else 0.8
        utility = (
            raw_weights["alpha_R"] * s_t.R
            + raw_weights["beta_M"] * s_t.M
            + raw_weights["gamma_T"] * s_t.T
            + raw_weights["delta_C"] * s_t.C
            + (raw_weights["lambda_H"] if raw_weights["lambda_H"] > 10.0 else raw_weights["lambda_H"] * h_val)
        )
        utility = round(utility, 4)

        # 3. MDP Policy Solver for PACE State Selection
        target_pace_state, reward = self.mdp_solver.solve_optimal_transition(
            s_t.pace_state, s_t, mia.threat_adjusted_utilities
        )

        # 4. Action Selection & Countermeasure Mapping (Supports legacy and new action keys)
        actions = []
        if "RF_JAMMING_SUSPECTED" in s_t.active_threats or "RF_JAMMING_DISRUPTION" in s_t.active_threats or s_t.C < 0.5 or s_t.Q < 0.5:
            actions.append("ACTIVATE_FREQUENCY_HOPPING")
            actions.append("TRIGGER_FREQUENCY_HOPPING")
        if "GPS_SPOOFING_SUSPECTED" in s_t.active_threats or "GPS_SPOOFING_CORRUPTION" in s_t.active_threats or s_t.T < 0.5:
            actions.append("SWITCH_TO_INERTIAL_NAV_FALLBACK")
            actions.append("REVERT_TO_IMU_NAV")
        if "RESOURCE_DOS_SUSPECTED" in s_t.active_threats or cpu_load > 0.70 or s_t.E < 0.30:
            actions.append("ENFORCE_PROCESS_QUOTA_ISOLATION")
            actions.append("KILL_DoS_PROCESS")

        # Deduplicate actions preserving order
        actions = list(dict.fromkeys(actions))

        # 5. Assistance Feasibility Score (AFS) Workload Bidding Offload check
        if cpu_load > 0.70 or "RESOURCE_DOS_SUSPECTED" in s_t.active_threats:
            best_peer, afs_score = self.afs_engine.evaluate_afs_bidding(local_cpu_load=cpu_load)
            if best_peer:
                actions.append("MIGRATE_TASK")
                logger.info("[Stage 6 AFS Bidding] Offloading heavy workload to %s (AFS Score: %.4f)", best_peer, afs_score)

        if not actions:
            actions.append("MAINTAIN_NOMINAL_OPERATIONS")

        # 6. Dynamic Redundancy Efficiency Index (DREI) calculation
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
            weights_applied=raw_weights,
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


# Backward compatibility aliases
HierarchicalOptimizationSolver = Stage6ResilienceOptimizer
Stage6Optimization = Stage6ResilienceOptimizer
