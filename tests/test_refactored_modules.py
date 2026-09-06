"""
Integration & Unit Test Suite for 5 Refactored HCCRO Modules
"""

import unittest
from src.core.interfaces import StateVectorData, PaceState
from src.hierarchy.satellite_node import SatelliteNode
from src.hierarchy.cluster_proxy import ClusterProxy
from src.hierarchy.constellation_manager import ConstellationManager
from src.hierarchy.ground_station import GroundStationNode
from src.optimization.constrained_solver import ConstrainedResilienceSolver
from src.optimization.pareto_frontier import ParetoFrontierGenerator
from src.stages.stage1_csa import SlidingWindowTelemetryParser
from src.stages.stage3_aim import BayesianIntentInference
from src.stages.stage7_healing import SelfHealingActuator
from src.stages.stage8_cke import CognitiveKnowledgeEngine
from src.validation.ablation_study import AblationFramework
from src.validation.baselines import HeuristicRuleBaseline, UnconstrainedBaseline, FlatNonHierarchicalBaseline
from src.validation.experiment_runner import ExperimentRunner


class TestRefactoredModules(unittest.TestCase):

    def test_module1_hierarchy_flow(self):
        ground_station = GroundStationNode("GS_TEST")
        manager = ConstellationManager("CONSTELLATION_TEST")
        cluster = ClusterProxy("CLUSTER_01")
        
        sat1 = SatelliteNode("SAT_01")
        sat2 = SatelliteNode("SAT_02")
        cluster.add_node(sat1)
        cluster.add_node(sat2)
        manager.add_cluster(cluster)

        out = manager.step_constellation()
        self.assertIn("global_pace_state", out)
        self.assertEqual(out["constellation_digest"]["total_satellites"], 2)

        policy = ground_station.issue_policy_update({"BATTERY_THRESHOLD": 0.35})
        self.assertEqual(policy["signature"], "RSA4096_AIRGAPPED_VERIFIED")

    def test_module2_constrained_slsqp_and_pareto(self):
        solver = ConstrainedResilienceSolver()
        s_t = StateVectorData(C=0.8, R=0.7, T=0.9, E=0.8, M=0.8)
        res = solver.solve(s_t, e_avail=0.7)
        self.assertIn("p_comm", res.x_opt)
        self.assertTrue(res.utility_value > 0.0)

        pareto_gen = ParetoFrontierGenerator(solver)
        frontier = pareto_gen.generate_frontier(s_t, num_samples=3)
        self.assertGreater(len(frontier), 0)

    def test_module3_cke_closed_loop_feedback(self):
        cke = CognitiveKnowledgeEngine()
        cke.log_incident("CYCLE_01", StateVectorData(), ["ACTIVATE_FREQUENCY_HOPPING"], PaceState.PRIMARY, 0.85, 0.90)
        
        success_rate = cke.query_action_success_rate("ACTIVATE_FREQUENCY_HOPPING")
        self.assertGreater(success_rate, 0.0)
        
        updated_probs = cke.compute_adaptive_transition_probs(["ACTIVATE_FREQUENCY_HOPPING"], "RF_JAMMING_DISRUPTION")
        self.assertIn("ACTIVATE_FREQUENCY_HOPPING", updated_probs)

    def test_module4_stage_gaps(self):
        # Sliding Window
        parser = SlidingWindowTelemetryParser(window_size=5)
        for i in range(5):
            parser.push({"snr_db": 10.0 + i})
        conf = parser.compute_anomaly_confidence()
        self.assertGreaterEqual(conf, 0.0)

        # Bayesian Inference
        bayes = BayesianIntentInference()
        scores = bayes.infer(["RF_JAMMING_SUSPECTED"])
        self.assertGreater(scores["RF_JAMMING_DISRUPTION"], 0.3)

        # Topology Actuators
        actuator = SelfHealingActuator()
        import networkx as nx
        g = nx.DiGraph()
        g.add_edge("SAT_01", "SAT_02")
        removed_edges = actuator.isolate_compromised_node(g, "SAT_01")
        self.assertGreater(len(removed_edges), 0)

    def test_module5_validation_runner(self):
        runner = ExperimentRunner(output_dir="results")
        summary = runner.run_benchmark_suite(num_runs=1, num_steps=3)
        self.assertIn("Full_4Layer_HCCRO", summary)
        self.assertGreater(summary["Full_4Layer_HCCRO"], 0.0)


if __name__ == "__main__":
    unittest.main()
