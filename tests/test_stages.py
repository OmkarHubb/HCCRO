"""
Stage-Level Unit Tests
======================
Verifies Stage 1 through Stage 8 output interfaces and algorithms.
"""

import unittest
from src.stages.stage1_csa import Stage1CSA
from src.stages.stage3_aim import Stage3AIM, PredictiveModelRegistry
from src.stages.stage6_optimization import Stage6Optimization, AdaptiveWeightController
from src.stages.stage8_cke import Stage8CKE
from src.core.interfaces import StateVectorData, CTIGOutput, PaceState
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

    def test_adaptive_weight_controller_critical_battery(self):
        # Battery at 0% -> delta_C = 0.0, lambda_H = 100.0
        weights_zero = AdaptiveWeightController.compute_weights(0.0)
        self.assertEqual(weights_zero["delta_C"], 0.0)
        self.assertEqual(weights_zero["lambda_H"], 100.0)

        # Battery at 30% (< 40%) -> delta_C quadratically reduced from 0.15
        weights_low = AdaptiveWeightController.compute_weights(0.30)
        self.assertLess(weights_low["delta_C"], 0.10)
        self.assertEqual(weights_low["lambda_H"], 100.0)

    def test_stage3_model_registry(self):
        registry = PredictiveModelRegistry()
        registry.register_model("RF_JAMMING_DISRUPTION", lambda ctig, st: 0.99)
        aim = Stage3AIM(registry=registry)

        s_t = StateVectorData()
        ctig = CTIGOutput(nx.DiGraph(), [], [], 0)
        output = aim.execute({"state_vector": s_t, "ctig_output": ctig})

        self.assertEqual(output.intention_scores["RF_JAMMING_DISRUPTION"], 0.99)

    def test_stage8_cke_persistence(self):
        cke = Stage8CKE()
        s_t = StateVectorData(active_threats=["RF_JAMMING_SUSPECTED"])
        output = cke.execute({"state_vector": s_t})
        self.assertTrue(output.persisted_successfully)


if __name__ == "__main__":
    unittest.main()
