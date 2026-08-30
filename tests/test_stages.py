"""
Stage-Level Unit Tests
======================
Verifies Stage 1 through Stage 8 output interfaces, algorithms, physical actuators, and persistent SQLite memory loop.
"""

import unittest
import time
from src.stages.stage1_csa import Stage1CSA
from src.stages.stage3_aim import Stage3AIM, PredictiveModelRegistry
from src.stages.stage6_optimization import (
    Stage6ResilienceOptimizer,
    Stage6Optimization,
    AdaptiveWeightController,
    PaceMdpSolver,
    AFSBiddingEngine,
)
from src.stages.stage7_healing import DistributedSelfHealing, Stage7Healing, SelfHealingActuator
from src.stages.stage8_cke import CyberKnowledgeEvolution, Stage8CKE
from src.core.interfaces import StateVectorData, CTIGOutput, PaceState, OptimizationOutput
import networkx as nx


class TestStages(unittest.TestCase):
    """Unit tests for individual stage components."""

    def test_stage1_csa_anomaly_detection(self):
        csa = Stage1CSA()
        raw_data = {
            "snr_db": 6.0,
            "pseudorange_error_m": 30.0,
            "cpu_usage_pct": 95.0,
            "energy_availability": 0.90,
        }
        state = csa.process(raw_data)

        self.assertIn("RF_JAMMING_SUSPECTED", state.active_threats)
        self.assertIn("GPS_SPOOFING_SUSPECTED", state.active_threats)
        self.assertIn("RESOURCE_DOS_SUSPECTED", state.active_threats)

    def test_adaptive_weight_controller_critical_battery_and_dcs_mos(self):
        # Battery at 0% -> delta_C = 0.0, lambda_H = 100.0
        weights_zero = AdaptiveWeightController.compute_weights(0.0)
        self.assertEqual(weights_zero["delta_C"], 0.0)
        self.assertEqual(weights_zero["lambda_H"], 100.0)

        # Battery at 30% (< 40%) -> delta_C quadratically reduced from 0.15
        weights_low = AdaptiveWeightController.compute_weights(0.30)
        self.assertLess(weights_low["delta_C"], 0.10)
        self.assertEqual(weights_low["lambda_H"], 100.0)

        # Test weight normalization
        norm = AdaptiveWeightController.get_normalized_weights(weights_low)
        self.assertAlmostEqual(sum(norm.values()), 1.0, delta=0.01)

    def test_pace_mdp_solver_emergency_zero_epsilon(self):
        solver = PaceMdpSolver(epsilon=0.5)  # Set high epsilon
        s_t = StateVectorData(E=0.10, active_threats=["RESOURCE_DOS_SUSPECTED"])
        
        # When in Emergency / critical low power, epsilon should be forced to 0.0
        target, reward = solver.solve_optimal_transition(PaceState.EMERGENCY, s_t, {})
        self.assertEqual(target, PaceState.EMERGENCY)

    def test_afs_bidding_engine(self):
        best_peer, score = AFSBiddingEngine.evaluate_afs_bidding(local_cpu_load=0.85)
        self.assertIsNotNone(best_peer)
        self.assertGreater(score, 0.5)

        # Low CPU load -> no offloading needed
        no_peer, score_zero = AFSBiddingEngine.evaluate_afs_bidding(local_cpu_load=0.40)
        self.assertIsNone(no_peer)
        self.assertEqual(score_zero, 0.0)

    def test_stage3_model_registry(self):
        registry = PredictiveModelRegistry()
        registry.register_model("RF_JAMMING_DISRUPTION", lambda ctig, st: 0.99)
        aim = Stage3AIM(registry=registry)

        s_t = StateVectorData()
        ctig = CTIGOutput(nx.DiGraph(), [], [], 0)
        output = aim.execute({"state_vector": s_t, "ctig_output": ctig})

        self.assertEqual(output.intention_scores["RF_JAMMING_DISRUPTION"], 0.99)

    def test_stage7_actuators_and_state_healing(self):
        actuator = SelfHealingActuator()
        s_t_damaged = StateVectorData(
            C=0.2, Q=0.2, T=0.3, E=0.2, active_threats=["RF_JAMMING_SUSPECTED", "GPS_SPOOFING_SUSPECTED", "RESOURCE_DOS_SUSPECTED"]
        )

        actions = ["TRIGGER_FREQUENCY_HOPPING", "REVERT_TO_IMU_NAV", "KILL_DoS_PROCESS"]
        healed_st = actuator.heal_state_vector(s_t_damaged, actions)

        self.assertEqual(healed_st.C, 1.0)
        self.assertEqual(healed_st.Q, 1.0)
        self.assertEqual(healed_st.T, 1.0)
        self.assertEqual(healed_st.E, 1.0)
        self.assertEqual(len(healed_st.active_threats), 0)

        # Execute healing stage
        stage7 = DistributedSelfHealing(actuator=actuator)
        opt_output = OptimizationOutput(
            optimal_actions=actions,
            target_pace_state=PaceState.PRIMARY,
            utility_score=0.95,
            drei_score=0.88,
            weights_applied={},
        )
        output = stage7.execute(opt_output)
        self.assertEqual(output.status, "SUCCESS")
        self.assertEqual(output.current_pace_state, PaceState.PRIMARY)

    def test_stage8_cke_persistence_and_logging(self):
        cke = CyberKnowledgeEvolution()
        timestamp = time.time()
        cycle_id = f"TEST_CYCLE_{int(timestamp)}"

        success = cke.log_incident_cycle(
            cycle_id=cycle_id,
            satellite_id="SAT_TEST_01",
            timestamp=timestamp,
            sensor_metrics={"cpu_usage": 85.0, "ram_usage": 40.0, "latency_ms": 120.0, "noise_floor_db": 15.0, "gps_drift_m": 5.0},
            attack_intent="RF_JAMMING_DISRUPTION",
            dcs_mos_weights={"alpha_R": 0.2, "beta_M": 0.2, "gamma_T": 0.2, "delta_C": 0.2, "lambda_H": 0.2},
            previous_pace_mode="PRIMARY",
            target_pace_mode="ALTERNATE",
            executed_action="TRIGGER_FREQUENCY_HOPPING",
            drei_score=0.85,
            mci_score=0.15,
            status="SUCCESS",
        )
        self.assertTrue(success)

        rate = cke.get_historical_success_rate("RF_JAMMING_DISRUPTION", "TRIGGER_FREQUENCY_HOPPING")
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

        s_t = StateVectorData(active_threats=["RF_JAMMING_SUSPECTED"])
        output = cke.execute({"state_vector": s_t})
        self.assertTrue(output.persisted_successfully)


if __name__ == "__main__":
    unittest.main()
