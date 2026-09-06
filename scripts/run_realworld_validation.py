"""
Real-World Mendeley GNSS Dataset Validation Driver (Hour 21 Active Campaign)
=============================================================================
Runs all 3600 real-world temporal epochs from December 21, 2023 through the
full 4-Layer HCCRO hierarchy and compares performance against the Heuristic Rule Baseline.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
from typing import Dict, Any, List

from src.data.gnss_parser import RealWorldGNSSParser
from src.hierarchy.constellation_manager import ConstellationManager
from src.hierarchy.cluster_proxy import ClusterProxy
from src.hierarchy.satellite_node import SatelliteNode
from src.hierarchy.ground_station import GroundStationNode
from src.validation.baselines import HeuristicRuleBaseline
from src.utils.logger import get_logger

logger = get_logger("Validation.RealWorldRunner")


def run_realworld_validation(output_dir: str = "results") -> Dict[str, Any]:
    """
    Executes real-world dataset ingestion and cognitive hierarchy validation across all 3600 epochs.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("[RealWorldRunner] Initializing RealWorldGNSSParser...")
    
    parser = RealWorldGNSSParser()
    records = parser.load_and_parse()
    num_epochs = len(records)

    logger.info("[RealWorldRunner] Initializing 4-Layer Cognitive Hierarchy (15 Satellites across 3 Clusters)...")
    ground_station = GroundStationNode("GS_AIRGAPPED_ROOT")
    manager = ConstellationManager("HCCRO_REALWORLD_CONSTELLATION")

    for c_idx in range(1, 4):
        cluster = ClusterProxy(f"CLUSTER_0{c_idx}")
        for s_idx in range(1, 6):
            sat_id = f"SAT_C{c_idx}_{s_idx:02d}"
            node = SatelliteNode(sat_id, cluster_id=f"CLUSTER_0{c_idx}")
            cluster.add_node(node)
        manager.add_cluster(cluster)

    heuristic_baseline = HeuristicRuleBaseline()

    # Tracking metrics
    hccro_resilience_history = []
    heuristic_resilience_history = []
    pace_state_history = []
    actions_triggered_history = []
    isolated_nodes_history = []

    previous_pace = "PRIMARY"
    state_transitions_count = 0

    start_time = time.time()

    for t, rec in enumerate(records):
        # 1. Map raw parsed epoch into telemetry dictionary feed
        epoch_telemetry = {
            f"SAT_C{c_idx}_{s_idx:02d}": rec
            for c_idx in range(1, 4) for s_idx in range(1, 6)
        }
        
        # 2. Step 4-Layer HCCRO Hierarchy
        c_map = {f"CLUSTER_0{c_idx}": epoch_telemetry for c_idx in range(1, 4)}
        out = manager.step_constellation(telemetry_map=c_map)
        digest = out["constellation_digest"]
        
        ground_station.ingest_constellation_telemetry(digest)

        current_pace = digest["global_pace_state"]
        if current_pace != previous_pace:
            state_transitions_count += 1
            previous_pace = current_pace

        hccro_resilience_history.append(digest["global_avg_resilience"])
        pace_state_history.append(current_pace)

        # Collect actions triggered in this epoch
        epoch_actions = []
        for c_out in out["cluster_outputs"].values():
            for n_res in c_out["node_results"].values():
                act = n_res.get("mitigation_plan", None)
                if act and hasattr(act, "optimal_actions"):
                    epoch_actions.extend(act.optimal_actions)
        
        actions_triggered_history.extend(set(epoch_actions))

        # Count isolated nodes
        isolated_cnt = sum(len(c_out.get("isolated_nodes", [])) for c_out in out["cluster_outputs"].values())
        isolated_nodes_history.append(isolated_cnt)

        # 3. Step Heuristic Baseline for comparative benchmarking
        h_out = heuristic_baseline.execute(rec["S_t"])
        heuristic_resilience_history.append(h_out.utility_score)

    elapsed_sec = round(time.time() - start_time, 2)

    # Executive Summary Calculation
    dataset_summary = parser.get_summary()
    hccro_mean_r = round(sum(hccro_resilience_history) / num_epochs, 4)
    heuristic_mean_r = round(sum(heuristic_resilience_history) / num_epochs, 4)
    improvement_pct = round(((hccro_mean_r - heuristic_mean_r) / heuristic_mean_r) * 100.0, 2)

    unique_actions = list(dict.fromkeys(actions_triggered_history))

    validation_report = {
        "dataset_info": {
            "source": "Mendeley GNSS Dataset (Active Campaign Dec 21, 2023 - Hour 21)",
            "location": "Science Hall of Yunnan University (25.0575 N, 102.6990 E)",
            "total_temporal_epochs": num_epochs,
            "execution_time_seconds": elapsed_sec,
        },
        "physical_telemetry_analytics": {
            "peak_coordinate_drift_meters": dataset_summary["peak_drift_meters"],
            "min_communication_quality_Q_t": dataset_summary["min_q_t"],
            "mean_communication_quality_Q_t": dataset_summary["mean_q_t"],
            "min_trust_state_T_t": dataset_summary["min_t_t"],
            "mean_trust_state_T_t": dataset_summary["mean_t_t"],
            "active_threat_epochs": dataset_summary["active_threat_epochs"],
        },
        "hccro_cognitive_performance": {
            "mean_resilience_score_R": hccro_mean_r,
            "total_autonomous_pace_state_transitions": state_transitions_count,
            "total_isolated_nodes_triggered": sum(isolated_nodes_history),
            "unique_self_healing_actuators_triggered": unique_actions,
        },
        "comparative_benchmark": {
            "hccro_4layer_mean_resilience": hccro_mean_r,
            "heuristic_rule_baseline_mean_resilience": heuristic_mean_r,
            "relative_improvement_percentage": f"{improvement_pct:+.2f}%",
        }
    }

    # Print Clean Validation Report to Console
    print("\n" + "=" * 80)
    print("      REAL-WORLD MENDELEY GNSS DATASET VALIDATION REPORT (HOUR 21 CAMPAIGN)")
    print("=" * 80)
    print(f" Dataset Source        : {validation_report['dataset_info']['source']}")
    print(f" Location              : {validation_report['dataset_info']['location']}")
    print(f" Epochs Processed      : {num_epochs} seconds ({elapsed_sec}s runtime)")
    print("-" * 80)
    print(" PHYSICAL TELEMETRY ANALYTICS (STAGE 1 CSA MAP)")
    print(f"   - Peak Coordinate Drift (meters)  : {dataset_summary['peak_drift_meters']:.4f} m")
    print(f"   - Min Comm Quality (Q_t)          : {dataset_summary['min_q_t']:.4f}  (Mean: {dataset_summary['mean_q_t']:.4f})")
    print(f"   - Min Trust State (T_t)           : {dataset_summary['min_t_t']:.4f}  (Mean: {dataset_summary['mean_t_t']:.4f})")
    print(f"   - Active Threat Detection Window  : {dataset_summary['active_threat_epochs']} / {num_epochs} epochs")
    print("-" * 80)
    print(" COGNITIVE HIERARCHY PERFORMANCE & MITIGATION")
    print(f"   - Mean Resilience Score (R)       : {hccro_mean_r:.4f}")
    print(f"   - Autonomous PACE Transitions     : {state_transitions_count}")
    print(f"   - Total Node Isolation Triggers  : {sum(isolated_nodes_history)}")
    print(f"   - Self-Healing Actuators Activated: {unique_actions}")
    print("-" * 80)
    print(" COMPARATIVE ARCHITECTURAL BENCHMARK")
    print(f"   - Full 4-Layer HCCRO Resilience   : {hccro_mean_r:.4f}")
    print(f"   - Heuristic Rule Baseline         : {heuristic_mean_r:.4f}")
    print(f"   - Relative Improvement            : {improvement_pct:+.2f}%")
    print("=" * 80 + "\n")

    # Export report to JSON
    json_out_path = os.path.join(output_dir, "realworld_validation_report.json")
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)
    logger.info("[RealWorldRunner] Saved validation report to %s", json_out_path)

    return validation_report


if __name__ == "__main__":
    run_realworld_validation()
