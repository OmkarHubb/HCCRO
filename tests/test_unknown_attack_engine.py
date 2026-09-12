"""
Unit Tests for Unknown-Attack Adaptive Response Engine
======================================================
Verifies open-set uncertainty detection, 12D behavioral fingerprinting,
counterfactual what-if evaluation, CKE knowledge persistence, and k-NN historical retrieval.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from src.core.interfaces import StateVectorData
from src.engine.unknown_attack_engine import UnknownAttackAdaptiveEngine


class TestUnknownAttackAdaptiveEngine(unittest.TestCase):

    def setUp(self):
        """Sets up a temporary directory and isolated SQLite database for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_cke_database.db")
        self.engine = UnknownAttackAdaptiveEngine(db_path=self.db_path, tau_unknown=0.45)

    def tearDown(self):
        """Cleans up temporary directory and database."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_1_open_set_threat_detection_and_fingerprint(self):
        """
        Test 1: Verify synthetic zero-day triggers 'UNKNOWN_ZERO_DAY_ATTACK'
        and creates a 12-dimensional fingerprint vector F_t.
        """
        # Synthetic low confidence probability distribution (max prob = 0.35 -> O_t = 0.65 >= 0.45)
        prob_dist = {
            "RF_JAMMING_DISRUPTION": 0.35,
            "GPS_SPOOFING_INJECTION": 0.25,
            "PROTOCOL_TELECOMMAND_INJECTION": 0.20,
        }

        # Degraded state vector
        state_vector = StateVectorData(
            C=0.40, R=0.50, T=0.45, Q=0.60, M=0.50, E=0.45, A=0.50
        )

        # 1. Detect open-set zero-day threat
        is_unknown = self.engine.detect_open_set_threat(prob_dist, state_vector)
        self.assertTrue(is_unknown, "Expected zero-day threat to be flagged as UNKNOWN_ZERO_DAY_ATTACK")

        # 2. Build 12D behavioral fingerprint F_t
        telemetry = {
            "processor_utilization": 85.0,
            "packet_delivery_ratio": 0.40,
            "communication_latency": 350.0,
            "navigation_consistency": 4.5,
            "memory_consumption": 75.0,
        }
        f_t = self.engine.build_fingerprint(state_vector, telemetry)

        self.assertIsInstance(f_t, np.ndarray)
        self.assertEqual(len(f_t), 12, "Fingerprint vector F_t must be exactly 12-dimensional")
        self.assertGreater(f_t[0], 0.0, "Delta C should reflect degradation")
        self.assertGreater(f_t[7], 0.0, "Delta CPU should reflect processor utilization spike")

    def test_2_counterfactual_evaluation_and_experience_persistence(self):
        """
        Test 2: Verify counterfactual evaluation selects safe response and
        persists experience into `cke_database.db` under table `unknown_attack_memory`.
        """
        state_vector = StateVectorData(
            C=0.30, R=0.40, T=0.35, Q=0.50, M=0.40, E=0.70, A=0.45
        )

        telemetry = {
            "processor_utilization": 80.0,
            "packet_delivery_ratio": 0.30,
            "communication_latency": 400.0,
            "navigation_consistency": 5.0,
            "memory_consumption": 80.0,
        }

        # Build fingerprint & search CKE memory (initially empty, returns candidate pool)
        f_t = self.engine.build_fingerprint(state_vector, telemetry)
        candidates = self.engine.search_cke_memory(f_t)

        self.assertIn("HYBRID_SHIELD", candidates)
        self.assertIn("PURGE_AND_RESET", candidates)

        # Counterfactual evaluation selects optimal action
        optimal_action = self.engine.evaluate_counterfactuals(state_vector, candidates)
        self.assertIsInstance(optimal_action, str)
        self.assertIn(optimal_action, candidates)

        # Persist experience into SQLite
        success = self.engine.persist_experience(
            fingerprint=f_t,
            action=optimal_action,
            mci_gain=0.35,
            cri_gain=0.42,
            success=True
        )
        self.assertTrue(success, "Experience persistence into unknown_attack_memory failed")

    def test_3_knn_historical_retrieval(self):
        """
        Test 3: Run second pass with similar fingerprint signature and confirm
        retrieval from CKE memory with Cosine Similarity Sim >= 0.80.
        """
        state_vector_1 = StateVectorData(
            C=0.30, R=0.40, T=0.35, Q=0.50, M=0.40, E=0.70, A=0.45
        )
        telemetry_1 = {
            "processor_utilization": 80.0,
            "packet_delivery_ratio": 0.30,
            "communication_latency": 400.0,
            "navigation_consistency": 5.0,
            "memory_consumption": 80.0,
        }

        # First pass: Build fingerprint & persist successful action 'HYBRID_SHIELD'
        f_t_1 = self.engine.build_fingerprint(state_vector_1, telemetry_1)
        self.engine.persist_experience(
            fingerprint=f_t_1,
            action="HYBRID_SHIELD",
            mci_gain=0.40,
            cri_gain=0.45,
            success=True
        )

        # Second pass: Very similar threat signature
        state_vector_2 = StateVectorData(
            C=0.32, R=0.42, T=0.36, Q=0.52, M=0.42, E=0.68, A=0.46
        )
        telemetry_2 = {
            "processor_utilization": 78.0,
            "packet_delivery_ratio": 0.32,
            "communication_latency": 390.0,
            "navigation_consistency": 4.8,
            "memory_consumption": 78.0,
        }
        f_t_2 = self.engine.build_fingerprint(state_vector_2, telemetry_2)

        # Calculate similarity manually to verify >= 0.80
        sim = float(np.dot(f_t_1, f_t_2) / (np.linalg.norm(f_t_1) * np.linalg.norm(f_t_2)))
        self.assertGreaterEqual(sim, 0.80, f"Expected Cosine Similarity >= 0.80, got {sim:.4f}")

        # Search CKE memory with f_t_2
        retrieved_actions = self.engine.search_cke_memory(f_t_2)

        self.assertIn("HYBRID_SHIELD", retrieved_actions, "k-NN search failed to retrieve historical action")


if __name__ == "__main__":
    unittest.main()
