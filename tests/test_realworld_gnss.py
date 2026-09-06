"""
Unit & Integration Test Suite for Real-World Mendeley GNSS Dataset Ingestion
"""

import unittest
from src.data.gnss_parser import RealWorldGNSSParser
from src.stages.stage1_csa import CyberSituationAwareness
from src.hierarchy.constellation_manager import ConstellationManager
from src.hierarchy.cluster_proxy import ClusterProxy
from src.hierarchy.satellite_node import SatelliteNode
from src.hierarchy.ground_station import GroundStationNode
from src.core.interfaces import StateVectorData, PaceState


class TestRealWorldGNSS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.parser = RealWorldGNSSParser()
        cls.records = cls.parser.load_and_parse()

    def test_gnss_parser_loading(self):
        self.assertEqual(len(self.records), 3600)
        df = self.parser.to_dataframe()
        self.assertEqual(len(df), 3600)
        self.assertIn("drift_meters", df.columns)
        self.assertIn("Q_t", df.columns)
        self.assertIn("T_t", df.columns)

    def test_mathematical_translation_formulas(self):
        # Epoch 0 should have drift ~0.0
        rec0 = self.records[0]
        self.assertAlmostEqual(rec0["drift_meters"], 0.0, delta=0.5)
        self.assertGreaterEqual(rec0["Q_t"], 0.0)
        self.assertLessEqual(rec0["Q_t"], 1.0)
        self.assertGreaterEqual(rec0["T_t"], 0.0)
        self.assertLessEqual(rec0["T_t"], 1.0)

        # Epoch with lowest T_t should trigger GPS_SPOOFING_SUSPECTED
        min_t_rec = min(self.records, key=lambda r: r["T_t"])
        self.assertLess(min_t_rec["T_t"], 0.5)
        self.assertIn("GPS_SPOOFING_SUSPECTED", min_t_rec["active_threats"])

    def test_realworld_stage1_csa_ingestion(self):
        csa = CyberSituationAwareness()
        epoch_rec = self.records[100]
        s_t = csa.execute(epoch_rec)
        self.assertIsInstance(s_t, StateVectorData)
        self.assertGreaterEqual(s_t.Q, 0.0)
        self.assertLessEqual(s_t.Q, 1.0)

    def test_realworld_hierarchy_propagation(self):
        manager = ConstellationManager("REALWORLD_TEST_CONSTELLATION")
        cluster = ClusterProxy("CLUSTER_01")
        sat1 = SatelliteNode("SAT_C1_01")
        sat2 = SatelliteNode("SAT_C1_02")
        cluster.add_node(sat1)
        cluster.add_node(sat2)
        manager.add_cluster(cluster)

        # Step 5 epochs of real dataset
        for t in range(5):
            epoch_rec = self.records[t]
            telem_map = {"CLUSTER_01": {"SAT_C1_01": epoch_rec, "SAT_C1_02": epoch_rec}}
            out = manager.step_constellation(telemetry_map=telem_map)
            self.assertIn("global_pace_state", out)
            self.assertIn("constellation_digest", out)


if __name__ == "__main__":
    unittest.main()
