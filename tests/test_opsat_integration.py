"""
OPSSAT-AD Integration Tests
============================
End-to-end integration test validating the ESA OPSSAT-AD Malicious Telecommand
Injection & Protocol Anomaly campaign dataset through the full 8-stage HCCRO pipeline.

Tests:
1. RealWorldOPSSATParser loads and translates data/processed/dataset.csv correctly.
2. 50 sequential segments pass through the orchestrator without exceptions.
3. Anomaly segments trigger correct S_t degradation, intent classification, weight
   shifts, actuator dispatch, and post-healing recovery.
"""

import unittest
import os
import sys

# Ensure project root is on sys.path for import resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.opsat_parser import RealWorldOPSSATParser
from src.core.orchestrator import HCCROOrchestrator
from src.core.interfaces import StateVectorData, PaceState
from src.stages.stage7_healing import SelfHealingActuator


class TestOPSSATParserUnit(unittest.TestCase):
    """Unit tests for the RealWorldOPSSATParser state translation engine."""

    def test_nominal_segment_produces_full_health(self):
        """anomaly==0 with benign features should produce near-nominal S_t."""
        s_t = RealWorldOPSSATParser.compute_state_vector(
            anomaly=0,
            diff_peaks=5.0,
            var_div_duration=1e-12,
            std_val=2e-5,
            gaps_squared=300.0,
            channel="CADC0872",
        )
        self.assertGreater(s_t.C, 0.9, "Nominal C_t should be near 1.0")
        self.assertEqual(s_t.R, 1.0, "Nominal R_t should be 1.0")
        self.assertGreater(s_t.T, 0.9, "Nominal T_t should be near 1.0")
        self.assertGreater(s_t.A, 0.9, "Nominal A_t should be near 1.0")
        self.assertEqual(len(s_t.active_threats), 0, "No threats on nominal segment")

    def test_anomaly_segment_degrades_state(self):
        """anomaly==1 should drop A_t, T_t, C_t, E_t significantly."""
        s_t = RealWorldOPSSATParser.compute_state_vector(
            anomaly=1,
            diff_peaks=50.0,
            var_div_duration=1e-4,
            std_val=0.2,
            gaps_squared=500.0,
            channel="CADC0872",
        )
        self.assertLess(s_t.A, 0.50, "A_t should drop under anomaly")
        self.assertLess(s_t.T, 0.50, "T_t should drop under anomaly")
        self.assertEqual(s_t.C, 0.20, "C_t = 1.0 - 0.80 = 0.20 under anomaly")
        self.assertEqual(s_t.R, 0.30, "R_t = 0.30 under anomaly")
        self.assertIn("TELECOMMAND_INJECTION_SUSPECTED", s_t.active_threats)
        self.assertIn("CADC_ANOMALY_SUSPECTED", s_t.active_threats)

    def test_state_vector_keys_unchanged(self):
        """Verify the 7-D State Vector dictionary keys are exactly {C, R, T, Q, M, E, A}."""
        s_t = RealWorldOPSSATParser.compute_state_vector(
            anomaly=0, diff_peaks=1.0, var_div_duration=1e-13,
            std_val=1e-5, gaps_squared=200.0,
        )
        expected_keys = {"C", "R", "T", "Q", "M", "E", "A"}
        self.assertEqual(set(s_t.to_dict().keys()), expected_keys)

    def test_all_dimensions_bounded(self):
        """All S_t dimensions must be in [0.0, 1.0] even with extreme inputs."""
        for anomaly in [0, 1]:
            for dp in [0.0, 240.0]:
                for vdd in [0.0, 2e-3]:
                    for std_v in [0.0, 0.6]:
                        for gs in [0.0, 21000.0]:
                            s_t = RealWorldOPSSATParser.compute_state_vector(
                                anomaly=anomaly, diff_peaks=dp,
                                var_div_duration=vdd, std_val=std_v,
                                gaps_squared=gs,
                            )
                            for dim_name, dim_val in s_t.to_dict().items():
                                self.assertGreaterEqual(dim_val, 0.0,
                                    f"{dim_name}={dim_val} < 0.0 with anom={anomaly}")
                                self.assertLessEqual(dim_val, 1.0,
                                    f"{dim_name}={dim_val} > 1.0 with anom={anomaly}")

    def test_m_t_equals_e_times_a(self):
        """M_t must equal clamp(E_t * A_t)."""
        s_t = RealWorldOPSSATParser.compute_state_vector(
            anomaly=1, diff_peaks=30.0, var_div_duration=5e-12,
            std_val=1e-4, gaps_squared=800.0,
        )
        expected_m = round(max(0.0, min(1.0, s_t.E * s_t.A)), 4)
        self.assertAlmostEqual(s_t.M, expected_m, places=4)


class TestOPSSATParserDatasetLoad(unittest.TestCase):
    """Tests that require the actual dataset.csv file."""

    @classmethod
    def setUpClass(cls):
        cls.parser = RealWorldOPSSATParser()
        cls.parser.load_and_parse()

    def test_dataset_loaded(self):
        """Parser should load all 2,123 segments."""
        self.assertGreater(len(self.parser.parsed_records), 2000)

    def test_summary_statistics(self):
        """Summary should report correct anomaly vs nominal split."""
        summary = self.parser.get_summary()
        self.assertGreater(summary["anomaly_segments"], 400)
        self.assertGreater(summary["nominal_segments"], 1600)
        self.assertEqual(
            summary["anomaly_segments"] + summary["nominal_segments"],
            summary["total_segments"],
        )

    def test_get_state_vector_returns_dataclass(self):
        """get_state_vector should return StateVectorData."""
        s_t = self.parser.get_state_vector(0)
        self.assertIsInstance(s_t, StateVectorData)

    def test_no_key_errors_across_dataset(self):
        """No KeyError or ZeroDivisionError across all segments."""
        for i in range(len(self.parser.parsed_records)):
            record = self.parser.parsed_records[i]
            s_t = record["S_t"]
            # Verify all 7 dimensions present
            d = s_t.to_dict()
            for key in ["C", "R", "T", "Q", "M", "E", "A"]:
                self.assertIn(key, d, f"Missing key {key} at segment {i}")


class TestOPSSATPipelineIntegration(unittest.TestCase):
    """End-to-end integration: 50 segments through the full 8-stage pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.parser = RealWorldOPSSATParser()
        cls.parser.load_and_parse()
        cls.orchestrator = HCCROOrchestrator()

    def test_50_segments_no_exceptions(self):
        """50 sequential segments pass through the pipeline without errors."""
        for i in range(50):
            record = self.parser.parsed_records[i]
            s_t = record["S_t"]

            # Feed through pipeline as a direct StateVectorData pass-through
            telemetry_input = {"S_t": s_t}
            try:
                result = self.orchestrator.execute_pipeline(telemetry_input)
            except (KeyError, ZeroDivisionError, TypeError) as exc:
                self.fail(
                    f"Pipeline raised {type(exc).__name__} on segment {i}: {exc}"
                )

            self.assertIn("state_vector", result)
            self.assertIn("aim_output", result)
            self.assertIn("mitigation_plan", result)
            self.assertIn("execution_status", result)
            self.assertIn("knowledge_record", result)

    def test_anomaly_segments_trigger_correct_intent(self):
        """When anomaly==1, Stage 3 should classify PROTOCOL_TELECOMMAND_INJECTION."""
        anomaly_records = [
            r for r in self.parser.parsed_records[:50] if r["anomaly"] == 1
        ]
        self.assertGreater(len(anomaly_records), 0, "No anomaly segments in first 50")

        for record in anomaly_records[:5]:  # Test first 5 anomaly segments
            s_t = record["S_t"]
            result = self.orchestrator.execute_pipeline({"S_t": s_t})

            aim_out = result["aim_output"]
            # PROTOCOL_TELECOMMAND_INJECTION should have highest score
            self.assertEqual(
                aim_out.primary_intention,
                "PROTOCOL_TELECOMMAND_INJECTION",
                f"Expected PROTOCOL_TELECOMMAND_INJECTION as primary intent, "
                f"got {aim_out.primary_intention} (scores: {aim_out.intention_scores})",
            )

    def test_anomaly_triggers_weight_shifts(self):
        """Anomaly segments with degraded A_t/T_t should trigger DCS-MOS weight overrides."""
        anomaly_records = [
            r for r in self.parser.parsed_records[:50] if r["anomaly"] == 1
        ]
        self.assertGreater(len(anomaly_records), 0)

        for record in anomaly_records[:3]:
            s_t = record["S_t"]
            result = self.orchestrator.execute_pipeline({"S_t": s_t})

            opt_out = result["mitigation_plan"]
            weights = opt_out.weights_applied

            # When A_t < 0.50 or T_t < 0.60 and TELECOMMAND_INJECTION_SUSPECTED:
            if s_t.A < 0.50 or s_t.T < 0.60:
                self.assertAlmostEqual(
                    weights["delta_C"], 0.05, places=2,
                    msg="delta_C should be suppressed to 0.05",
                )
                self.assertAlmostEqual(
                    weights["gamma_T"], 0.80, places=2,
                    msg="gamma_T should be elevated to 0.80",
                )
                self.assertAlmostEqual(
                    weights["lambda_H"], 0.55, places=2,
                    msg="lambda_H should spike to 0.55",
                )

    def test_anomaly_triggers_correct_actuators(self):
        """Stage 7 should dispatch PURGE_MALICIOUS_APID_QUEUE and REVERT_TO_SAFE_TC_KEY_STORE."""
        anomaly_records = [
            r for r in self.parser.parsed_records[:50] if r["anomaly"] == 1
        ]
        self.assertGreater(len(anomaly_records), 0)

        for record in anomaly_records[:3]:
            s_t = record["S_t"]
            result = self.orchestrator.execute_pipeline({"S_t": s_t})

            heal_out = result["execution_status"]
            self.assertIn(
                "PURGE_MALICIOUS_APID_QUEUE", heal_out.executed_actions,
                "PURGE_MALICIOUS_APID_QUEUE should be in executed actions",
            )
            self.assertIn(
                "REVERT_TO_SAFE_TC_KEY_STORE", heal_out.executed_actions,
                "REVERT_TO_SAFE_TC_KEY_STORE should be in executed actions",
            )

    def test_healing_restores_state_vector(self):
        """After healing actuators execute, S_t should restore to nominal baselines."""
        anomaly_records = [
            r for r in self.parser.parsed_records[:50] if r["anomaly"] == 1
        ]
        self.assertGreater(len(anomaly_records), 0)

        actuator = SelfHealingActuator()

        for record in anomaly_records[:3]:
            s_t = record["S_t"]
            # Verify initial degradation
            self.assertLess(s_t.A, 1.0, "A_t should be degraded before healing")
            self.assertLess(s_t.T, 1.0, "T_t should be degraded before healing")

            # Apply healing
            healed = actuator.heal_state_vector(
                s_t, ["PURGE_MALICIOUS_APID_QUEUE", "REVERT_TO_SAFE_TC_KEY_STORE"]
            )
            self.assertEqual(healed.C, 1.0, "C_t should restore to 1.0")
            self.assertEqual(healed.A, 1.0, "A_t should restore to 1.0")
            self.assertEqual(healed.T, 1.0, "T_t should restore to 1.0")
            self.assertEqual(healed.E, 1.0, "E_t should restore to 1.0")
            # Threats should be cleared
            self.assertNotIn("TELECOMMAND_INJECTION_SUSPECTED", healed.active_threats)
            self.assertNotIn("CADC_ANOMALY_SUSPECTED", healed.active_threats)

    def test_nominal_segments_stay_healthy(self):
        """Nominal segments (anomaly==0) should NOT trigger telecommand injection."""
        nominal_records = [
            r for r in self.parser.parsed_records[:50] if r["anomaly"] == 0
        ]
        self.assertGreater(len(nominal_records), 0)

        for record in nominal_records[:5]:
            s_t = record["S_t"]
            self.assertNotIn("TELECOMMAND_INJECTION_SUSPECTED", s_t.active_threats)
            self.assertEqual(s_t.R, 1.0)
            self.assertEqual(s_t.C, 1.0)

    def test_cke_logging_persists(self):
        """Stage 8 CKE should successfully persist incident records."""
        record = self.parser.parsed_records[0]
        s_t = record["S_t"]
        result = self.orchestrator.execute_pipeline({"S_t": s_t})

        cke_out = result["knowledge_record"]
        self.assertTrue(
            cke_out.persisted_successfully,
            "CKE should persist the record successfully",
        )
        self.assertIsNotNone(cke_out.record_id)


if __name__ == "__main__":
    unittest.main()
