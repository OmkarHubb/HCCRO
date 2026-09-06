"""
Module 1: Layer 2 — Cluster Proxy Agent (ClusterProxy)
=====================================================
Manages a localized cluster of satellite nodes (e.g. 5 satellites per cluster).
Executes inter-node consensus, AFS workload bidding, intra-cluster graph topology,
and local fault isolation.
"""

from typing import Dict, Any, List, Optional
import networkx as nx
from src.hierarchy.satellite_node import SatelliteNode
from src.stages.stage6_optimization import AFSBiddingEngine
from src.utils.logger import get_logger

logger = get_logger("Hierarchy.ClusterProxy")


class ClusterProxy:
    """
    Layer 2 Agent: Cluster Proxy Manager.
    """

    def __init__(self, cluster_id: str, consensus_threshold: float = 0.60):
        self.cluster_id = cluster_id
        self.consensus_threshold = consensus_threshold
        self.nodes: Dict[str, SatelliteNode] = {}
        self.cluster_graph: nx.DiGraph = nx.DiGraph()

    def add_node(self, node: SatelliteNode) -> None:
        """Registers a satellite node to this cluster and updates cluster topology graph."""
        node.cluster_id = self.cluster_id
        self.nodes[node.node_id] = node
        self.cluster_graph.add_node(node.node_id, status=node.status)
        
        # Connect to existing cluster nodes (mesh topology)
        for existing_id in self.nodes:
            if existing_id != node.node_id:
                self.cluster_graph.add_edge(node.node_id, existing_id, latency=0.05, weight=1.0)
                self.cluster_graph.add_edge(existing_id, node.node_id, latency=0.05, weight=1.0)
        logger.info("[%s] Registered Satellite Node %s. Cluster size: %d", self.cluster_id, node.node_id, len(self.nodes))

    def step_cluster(self, cluster_telemetry: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Executes one cognitive cycle across all member satellite nodes in the cluster.
        
        Args:
            cluster_telemetry: Optional dict mapping node_id -> telemetry dict.
            
        Returns:
            Dict containing cluster-wide execution results, consensus results, and AFS offloading log.
        """
        node_results = {}
        digests = []

        # 1. Step individual nodes
        for node_id, node in self.nodes.items():
            telem = cluster_telemetry.get(node_id) if cluster_telemetry else None
            res = node.step(telemetry=telem)
            node_results[node_id] = res
            digests.append(node.get_telemetry_digest())

        # 2. Execute intra-cluster Byzantine/Majority Consensus
        consensus = self.execute_consensus(digests)

        # 3. Inter-node Workload Bidding & Offloading (AFS)
        offload_log = self.manage_afs_bidding(digests)

        # 4. Local Fault Isolation
        isolated = self.isolate_compromised_nodes()

        return {
            "cluster_id": self.cluster_id,
            "node_results": node_results,
            "consensus": consensus,
            "offload_log": offload_log,
            "isolated_nodes": isolated,
            "cluster_digest": self.get_cluster_digest(),
        }

    def execute_consensus(self, digests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Performs Byzantine/Majority voting on cluster threat state.
        If > 60% of healthy nodes report a threat, escalates cluster threat consensus.
        """
        active_nodes = [d for d in digests if d["status"] != "ISOLATED"]
        if not active_nodes:
            return {"consensus_threat": "NONE", "consensus_confidence": 0.0}

        threat_counts: Dict[str, int] = {}
        for d in active_nodes:
            for threat in d["active_threats"]:
                threat_counts[threat] = threat_counts.get(threat, 0) + 1

        cluster_threats = []
        total_active = len(active_nodes)
        
        for threat, count in threat_counts.items():
            ratio = count / total_active
            if ratio >= self.consensus_threshold:
                cluster_threats.append(threat)
                logger.warning("[%s Consensus] Threat '%s' confirmed by majority vote (%.1f%% of nodes)!",
                               self.cluster_id, threat, ratio * 100)

        return {
            "consensus_threats": cluster_threats,
            "participating_nodes": total_active,
            "consensus_ratio": round(len(cluster_threats) / max(1, len(threat_counts)), 4) if threat_counts else 0.0,
        }

    def manage_afs_bidding(self, digests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluates AFS bids across cluster nodes to offload tasks from overloaded nodes."""
        offload_log = []
        
        overloaded = [d for d in digests if d["cpu_load"] > 0.70 and d["status"] != "ISOLATED"]
        healthy_peers = [
            {
                "id": d["node_id"],
                "battery_margin": d["state_vector"]["E"],
                "cpu_margin": max(0.0, 1.0 - d["cpu_load"]),
                "trust_score": d["state_vector"]["T"],
            }
            for d in digests if d["cpu_load"] <= 0.60 and d["status"] == "HEALTHY"
        ]

        if not healthy_peers:
            return offload_log

        for source in overloaded:
            best_peer, afs_score = AFSBiddingEngine.evaluate_afs_bidding(
                neighbors=healthy_peers, local_cpu_load=source["cpu_load"]
            )
            if best_peer:
                offload_log.append({
                    "source_node": source["node_id"],
                    "target_node": best_peer,
                    "afs_score": afs_score,
                })
                logger.info("[%s AFS Offload] Task migrated from %s -> %s (AFS Score: %.4f)",
                            self.cluster_id, source["node_id"], best_peer, afs_score)

        return offload_log

    def isolate_compromised_nodes(self) -> List[str]:
        """Identifies compromised nodes and removes them from cluster topology."""
        isolated = []
        for node_id, node in self.nodes.items():
            if node.status == "COMPROMISED" or node.state_vector.R < 0.25:
                node.isolate()
                isolated.append(node_id)
                # Drop from graph
                if self.cluster_graph.has_node(node_id):
                    self.cluster_graph.remove_node(node_id)
                logger.warning("[%s Cluster Isolation] Isolated node %s from cluster graph.", self.cluster_id, node_id)
        return isolated

    def get_cluster_digest(self) -> Dict[str, Any]:
        """Returns aggregated cluster health digest."""
        node_statuses = {n_id: n.status for n_id, n in self.nodes.items()}
        avg_r = (
            sum(n.state_vector.R for n in self.nodes.values()) / max(1, len(self.nodes))
        )
        return {
            "cluster_id": self.cluster_id,
            "total_nodes": len(self.nodes),
            "node_statuses": node_statuses,
            "avg_resilience_R": round(avg_r, 4),
            "active_graph_edges": len(self.cluster_graph.edges()),
        }
