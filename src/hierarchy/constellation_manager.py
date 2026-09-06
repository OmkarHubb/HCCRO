"""
Module 1: Layer 3 — Constellation Manager Agent (ConstellationManager)
========================================================================
Manages multiple satellite clusters (e.g. 3 clusters × 5 satellites = 15 total nodes).
Coordinates orbit-wide state aggregation, cross-cluster resource re-allocation,
and global constellation-wide PACE state escalations.
"""

from typing import Dict, Any, List, Optional
from src.core.interfaces import PaceState
from src.hierarchy.cluster_proxy import ClusterProxy
from src.utils.logger import get_logger

logger = get_logger("Hierarchy.ConstellationManager")


class ConstellationManager:
    """
    Layer 3 Agent: Constellation Manager.
    """

    def __init__(self, constellation_name: str = "HCCRO_CONSTELLATION_ALPHA"):
        self.constellation_name = constellation_name
        self.clusters: Dict[str, ClusterProxy] = {}
        self.global_pace_state = PaceState.PRIMARY
        self.constellation_history: List[Dict[str, Any]] = []

    def add_cluster(self, cluster: ClusterProxy) -> None:
        """Registers a satellite cluster with the constellation manager."""
        self.clusters[cluster.cluster_id] = cluster
        logger.info("[%s] Registered Cluster Proxy %s. Total clusters: %d",
                    self.constellation_name, cluster.cluster_id, len(self.clusters))

    def step_constellation(self, telemetry_map: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Executes one constellation-wide cognitive cycle across all clusters.
        
        Args:
            telemetry_map: Optional nested dict mapping cluster_id -> node_id -> telemetry dict.
            
        Returns:
            Dict containing constellation execution summary, global PACE state, and cluster digests.
        """
        cluster_outputs = {}
        
        for c_id, cluster in self.clusters.items():
            c_telem = telemetry_map.get(c_id) if telemetry_map else None
            cluster_outputs[c_id] = cluster.step_cluster(cluster_telemetry=c_telem)

        # Evaluate constellation-wide PACE state escalation
        new_pace = self.evaluate_global_pace_escalation(cluster_outputs)
        self.global_pace_state = new_pace

        # Cross-cluster resource re-allocation
        cross_alloc = self.reallocate_cross_cluster_resources(cluster_outputs)

        digest = self.get_constellation_digest()
        self.constellation_history.append(digest)

        return {
            "constellation_name": self.constellation_name,
            "global_pace_state": self.global_pace_state.value,
            "cluster_outputs": cluster_outputs,
            "cross_cluster_allocations": cross_alloc,
            "constellation_digest": digest,
        }

    def evaluate_global_pace_escalation(self, cluster_outputs: Dict[str, Any]) -> PaceState:
        """
        Evaluates global PACE state escalation based on constellation-wide health metrics:
        - If compromised/degraded node ratio > 40% or average resilience R < 0.35 -> EMERGENCY
        - If compromised/degraded node ratio > 20% or average resilience R < 0.55 -> CONTINGENCY
        - If active threats in multiple clusters -> ALTERNATE
        - Otherwise -> PRIMARY
        """
        total_sats = 0
        degraded_sats = 0
        resilience_sum = 0.0

        for c_id, c_out in cluster_outputs.items():
            digest = c_out["cluster_digest"]
            total_sats += digest["total_nodes"]
            resilience_sum += digest["avg_resilience_R"] * digest["total_nodes"]
            
            for n_id, status in digest["node_statuses"].items():
                if status in ["DEGRADED", "COMPROMISED", "ISOLATED"]:
                    degraded_sats += 1

        if total_sats == 0:
            return PaceState.PRIMARY

        global_avg_r = resilience_sum / total_sats
        failure_ratio = degraded_sats / total_sats

        if failure_ratio > 0.40 or global_avg_r < 0.35:
            escalated = PaceState.EMERGENCY
        elif failure_ratio > 0.20 or global_avg_r < 0.55:
            escalated = PaceState.CONTINGENCY
        elif failure_ratio > 0.0:
            escalated = PaceState.ALTERNATE
        else:
            escalated = PaceState.PRIMARY

        if escalated != self.global_pace_state:
            logger.warning("[%s] GLOBAL PACE ESCALATION: %s -> %s (Failure Ratio: %.1f%%, Avg R: %.4f)",
                           self.constellation_name, self.global_pace_state.value, escalated.value,
                           failure_ratio * 100, global_avg_r)

        return escalated

    def reallocate_cross_cluster_resources(self, cluster_outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Re-allocates comms/tasks across clusters if an entire cluster is degraded."""
        reallocations = []
        cluster_healths = {}
        
        for c_id, c_out in cluster_outputs.items():
            cluster_healths[c_id] = c_out["cluster_digest"]["avg_resilience_R"]

        sorted_clusters = sorted(cluster_healths.items(), key=lambda x: x[1])
        
        if len(sorted_clusters) >= 2:
            weakest_id, min_r = sorted_clusters[0]
            strongest_id, max_r = sorted_clusters[-1]
            
            if min_r < 0.40 and max_r > 0.70:
                reallocations.append({
                    "from_cluster": weakest_id,
                    "to_cluster": strongest_id,
                    "reallocated_bandwidth_mbps": 50.0,
                    "reason": f"Cluster {weakest_id} degraded (R={min_r:.2f}). Offloading traffic to {strongest_id} (R={max_r:.2f}).",
                })
                logger.info("[%s Cross-Cluster] Reallocating bandwidth from %s to %s",
                            self.constellation_name, weakest_id, strongest_id)

        return reallocations

    def get_constellation_digest(self) -> Dict[str, Any]:
        """Returns constellation-wide health summary."""
        total_nodes = sum(len(c.nodes) for c in self.clusters.values())
        all_resilience = [
            n.state_vector.R for c in self.clusters.values() for n in c.nodes.values()
        ]
        avg_r = sum(all_resilience) / max(1, len(all_resilience))

        return {
            "constellation_name": self.constellation_name,
            "global_pace_state": self.global_pace_state.value,
            "total_clusters": len(self.clusters),
            "total_satellites": total_nodes,
            "global_avg_resilience": round(avg_r, 4),
        }
