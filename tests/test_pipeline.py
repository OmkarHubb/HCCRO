"""
End-to-End Pipeline Unit Tests
==============================
Tests the master orchestrator execution loop across 5-tick scenarios, PACE transitions,
and Adaptive Weight Controller triggers.
"""

import unittest
from src.core.orchestrator import HCCROOrchestrator
from src.core.interfaces import PaceState


class TestPipeline(unittest.TestCase):
    """Unit tests for master orchestrator pipeline execution."""

    def test_orchestrator_nominal_run(self):
        orchestrator = HCCROOrchestrator()
        telemetry = {
            "snr_db": 25.0,
            "pseudorange_error_m": 2.0,
            "cpu_usage_pct": 20.0,
            "energy_availability": 0.98,
        }
        result = orchestrator.run_cycle(telemetry)
        self.assertIsNotNone(result)
        self.assertEqual(result["execution_status"]["status"], "SUCCESS")
        self.assertEqual(result["pace_state"], PaceState.PRIMARY.value)

    def test_orchestrator_jamming_run(self):
        orchestrator = HCCROOrchestrator()
        telemetry = {
            "snr_db": 5.0,  # RF Jamming
            "pseudorange_error_m": 2.0,
            "cpu_usage_pct": 20.0,
            "energy_availability": 0.95,
        }
        result = orchestrator.run_cycle(telemetry)
        self.assertIn("ACTIVATE_FREQUENCY_HOPPING", result["mitigation_plan"]["optimal_actions"])

    def test_orchestrator_dos_and_battery_critical_run(self):
        orchestrator = HCCROOrchestrator()
        telemetry = {
            "snr_db": 25.0,
            "pseudorange_error_m": 2.0,
            "cpu_usage_pct": 99.0,  # DoS
            "memory_usage_pct": 95.0,
            "energy_availability": 0.35,  # Battery < 40%!
        }
        res = orchestrator.execute_pipeline(telemetry)
        opt = res["mitigation_plan"]
        heal = res["execution_status"]

        # Check Adaptive Weight Controller response
        self.assertEqual(opt.weights_applied["delta_C"], 0.0)
        self.assertEqual(opt.weights_applied["lambda_H"], 100.0)
        self.assertEqual(heal.current_pace_state, PaceState.EMERGENCY)


if __name__ == "__main__":
    unittest.main()
