"""
Stage 2: Cognitive Threat Intelligence Graph (CTIG)
===================================================
Constructs and maintains a dynamic NetworkX multi-layer dependency graph representing
satellite buses, payload transceivers, inter-satellite links, and onboard software services.

Ingests Stage 1 Operational State Vectors (S_t) and raw telemetry feeds to inject dynamic
threat indicators, degrade cyber-physical edge weights under active attacks, and heal graph
topologies when threats resolve.

Hollow & Extensible Framework Design:
- Provides clean input/output interfaces (`execute` -> `CTIGOutput`).
- Scaffolds vertex types (SATELLITE, LINK, SERVICE, THREAT_INDICATOR).
- Scaffolds edge relations (DEPENDS_ON, COMMUNICATES_WITH, TRUSTS, EXPLOITS).
- Features comment-scaffolded # TODO slots for PageRank/Centrality, GNN embeddings, and Dijkstra attack paths.
"""

from typing import Dict, Any, Union, List, Optional, Tuple
import networkx as nx

from src.core.interfaces import BaseStage, StateVectorData, TelemetryData, CTIGOutput
from src.core.state_vector import StateVector
from src.utils.logger import get_logger

logger = get_logger("Stage2.CTIG")

# =============================================================================
# Domain Constants: Graph Vertices & Edge Relations
# =============================================================================
NODE_TYPE_SATELLITE = "SATELLITE"
NODE_TYPE_LINK = "LINK"
NODE_TYPE_SERVICE = "SERVICE"
NODE_TYPE_THREAT = "THREAT_INDICATOR"

EDGE_RELATION_DEPENDS_ON = "DEPENDS_ON"
EDGE_RELATION_COMMUNICATES_WITH = "COMMUNICATES_WITH"
EDGE_RELATION_TRUSTS = "TRUSTS"
EDGE_RELATION_EXPLOITS = "EXPLOITS"

THREAT_SUBTYPE_JAMMING = "RF_JAMMING"
THREAT_SUBTYPE_SPOOFING = "GNSS_SPOOFING"
THREAT_SUBTYPE_DOS = "RESOURCE_EXHAUSTION"


class CognitiveThreatIntelligenceGraph(BaseStage):
    """
    Stage 2: Cognitive Threat Intelligence Graph (CTIG) Engine.
    Inherits from BaseStage to enforce plug-and-play adaptability across the HCCRO pipeline.
    """

    def __init__(self, constellation_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the CTIG stage with a constellation layout configuration.
        
        Args:
            constellation_config: Optional configuration dictionary defining satellite nodes,
                                  ISL topology, and software services.
        """
        super().__init__(name="Stage2_CTIG")
        self.config = constellation_config or self._default_constellation_layout()

    def _default_constellation_layout(self) -> Dict[str, Any]:
        """Returns default constellation layout configuration for 3 LEO satellites."""
        return {
            "satellites": ["SAT_LEO_01", "SAT_LEO_02", "SAT_LEO_03"],
            "links": [
                {"id": "LINK_RF_01", "sat": "SAT_LEO_01", "type": "RF_TRANSCEIVER"},
                {"id": "LINK_RF_02", "sat": "SAT_LEO_02", "type": "RF_TRANSCEIVER"},
                {"id": "LINK_RF_03", "sat": "SAT_LEO_03", "type": "RF_TRANSCEIVER"},
                {"id": "LINK_ISL_01_02", "source": "SAT_LEO_01", "target": "SAT_LEO_02", "type": "ISL"},
                {"id": "LINK_ISL_02_03", "source": "SAT_LEO_02", "target": "SAT_LEO_03", "type": "ISL"},
            ],
            "services": ["SERVICE_NAV", "SERVICE_ROUTING", "SERVICE_TELEMETRY"],
        }

    def build_base_graph(self) -> nx.DiGraph:
        """
        Constructs the nominal, uncompromised dynamic networkx.DiGraph representing the
        cyber-physical constellation architecture.
        
        Returns:
            nx.DiGraph: Baseline network graph with multi-type nodes and edges.
        """
        graph = nx.DiGraph()

        # 1. Multi-Type Vertex Creation: SATELLITE nodes
        for sat in self.config["satellites"]:
            graph.add_node(
                sat,
                type=NODE_TYPE_SATELLITE,
                status="HEALTHY",
                cpu_load=0.15,
                ram_usage=0.20,
                battery_pct=98.0,
            )

        # 2. Multi-Type Vertex Creation: LINK nodes (Transceivers & ISLs)
        for link in self.config["links"]:
            link_id = link["id"]
            graph.add_node(
                link_id,
                type=NODE_TYPE_LINK,
                link_type=link["type"],
                status="HEALTHY",
                pdr=1.0,
                bandwidth_mbps=100.0,
            )
            # Link connections to Satellites
            if "sat" in link:
                graph.add_edge(link["sat"], link_id, relation=EDGE_RELATION_COMMUNICATES_WITH, weight=1.0)
                graph.add_edge(link_id, link["sat"], relation=EDGE_RELATION_COMMUNICATES_WITH, weight=1.0)
            elif "source" in link and "target" in link:
                graph.add_edge(link["source"], link_id, relation=EDGE_RELATION_COMMUNICATES_WITH, weight=1.0)
                graph.add_edge(link_id, link["target"], relation=EDGE_RELATION_COMMUNICATES_WITH, weight=1.0)

        # 3. Multi-Type Vertex Creation: SERVICE nodes (Per satellite)
        for sat in self.config["satellites"]:
            for svc_base in self.config["services"]:
                svc_id = f"{svc_base}_{sat}"
                graph.add_node(
                    svc_id,
                    type=NODE_TYPE_SERVICE,
                    service_name=svc_base,
                    parent_sat=sat,
                    status="HEALTHY",
                )
                # DEPENDS_ON edge: Software service maps to Satellite compute resources
                graph.add_edge(svc_id, sat, relation=EDGE_RELATION_DEPENDS_ON, weight=1.0)

        # 4. Multi-Type Edge Creation: TRUSTS edges (Peer trust propagation)
        sats = self.config["satellites"]
        for i in range(len(sats)):
            for j in range(i + 1, len(sats)):
                sat_a, sat_b = sats[i], sats[j]
                graph.add_edge(sat_a, sat_b, relation=EDGE_RELATION_TRUSTS, weight=1.0)
                graph.add_edge(sat_b, sat_a, relation=EDGE_RELATION_TRUSTS, weight=1.0)

        return graph

    def _graph_builder_heuristic_slot(
        self, s_t: StateVectorData, raw_telemetry: Optional[TelemetryData] = None
    ) -> CTIGOutput:
        """
        Dynamic Anomaly Mapping & Evolution Logic.
        Ingests the S_t state vector and raw telemetry to dynamically inject threat nodes,
        draw EXPLOITS edges, degrade edge weights, and execute graph healing when threats clear.
        
        Pluggable heuristic slot - can be augmented with PyTorch Geometric / Graph Neural Networks.
        """
        graph = self.build_base_graph()

        # Primary satellite under observation (defaults to SAT_LEO_01)
        target_sat = "SAT_LEO_01"
        target_link = "LINK_RF_01"
        target_nav_svc = f"SERVICE_NAV_{target_sat}"

        # Parse telemetry indicators
        pdr = raw_telemetry.packet_delivery_ratio if raw_telemetry else s_t.C
        spectrum_usage = raw_telemetry.spectrum_usage if raw_telemetry else (1.0 - s_t.C)
        nav_drift = raw_telemetry.navigation_consistency if raw_telemetry else (1.0 - s_t.T) * 20.0
        auth_events = raw_telemetry.authentication_events if raw_telemetry else (4 if s_t.T < 0.5 else 0)
        cpu_util = raw_telemetry.processor_utilization if raw_telemetry else (1.0 - s_t.E) * 100.0
        ram_util = raw_telemetry.memory_consumption if raw_telemetry else (1.0 - s_t.E) * 100.0

        active_threat_set = set(s_t.active_threats)

        # ---------------------------------------------------------------------
        # 1. ANOMALY MAPPING: RF Jamming
        # ---------------------------------------------------------------------
        is_jamming = (
            "RF_JAMMING_SUSPECTED" in active_threat_set
            or s_t.C < 0.5
            or pdr < 0.5
            or spectrum_usage > 0.5
        )
        if is_jamming:
            threat_node = f"THREAT_RF_JAMMING_{target_sat}"
            graph.add_node(
                threat_node,
                type=NODE_TYPE_THREAT,
                subtype=THREAT_SUBTYPE_JAMMING,
                severity="HIGH",
                status="ACTIVE",
                timestamp=s_t.timestamp,
            )
            # Inject EXPLOITS edge pointing to target satellite's LINK node
            graph.add_edge(threat_node, target_link, relation=EDGE_RELATION_EXPLOITS, weight=0.90)

            # Reduce communication edge weights connected to target_link and target_sat
            for u, v, data in graph.edges(data=True):
                if data.get("relation") == EDGE_RELATION_COMMUNICATES_WITH:
                    if target_link in (u, v) or target_sat in (u, v):
                        graph[u][v]["weight"] = round(graph[u][v]["weight"] * 0.20, 3)

            graph.nodes[target_link]["status"] = "COMPROMISED"
            logger.debug("[CTIG Evolution] Injected RF_JAMMING threat indicator on %s", target_link)

        # ---------------------------------------------------------------------
        # 2. ANOMALY MAPPING: GNSS Spoofing
        # ---------------------------------------------------------------------
        is_spoofing = (
            "GPS_SPOOFING_SUSPECTED" in active_threat_set
            or s_t.T < 0.5
            or nav_drift > 15.0
            or auth_events > 2
        )
        if is_spoofing:
            threat_node = f"THREAT_GNSS_SPOOFING_{target_sat}"
            graph.add_node(
                threat_node,
                type=NODE_TYPE_THREAT,
                subtype=THREAT_SUBTYPE_SPOOFING,
                severity="CRITICAL",
                status="ACTIVE",
                timestamp=s_t.timestamp,
            )
            # Inject EXPLOITS edge pointing to navigation SERVICE node
            graph.add_edge(threat_node, target_nav_svc, relation=EDGE_RELATION_EXPLOITS, weight=0.95)

            # Degrade surrounding TRUSTS edge weights
            for u, v, data in graph.edges(data=True):
                if data.get("relation") == EDGE_RELATION_TRUSTS:
                    if target_sat in (u, v):
                        graph[u][v]["weight"] = round(graph[u][v]["weight"] * 0.10, 3)

            graph.nodes[target_nav_svc]["status"] = "COMPROMISED"
            logger.debug("[CTIG Evolution] Injected GNSS_SPOOFING threat indicator on %s", target_nav_svc)

        # ---------------------------------------------------------------------
        # 3. ANOMALY MAPPING: Resource Exhaustion (DoS)
        # ---------------------------------------------------------------------
        is_dos = (
            "RESOURCE_DOS_SUSPECTED" in active_threat_set
            or s_t.E < 0.2
            or cpu_util > 85.0
            or ram_util > 90.0
        )
        if is_dos:
            threat_node = f"THREAT_RESOURCE_EXHAUSTION_{target_sat}"
            graph.add_node(
                threat_node,
                type=NODE_TYPE_THREAT,
                subtype=THREAT_SUBTYPE_DOS,
                severity="CRITICAL",
                status="ACTIVE",
                timestamp=s_t.timestamp,
            )
            # Inject EXPLOITS edge pointing to target SATELLITE node
            graph.add_edge(threat_node, target_sat, relation=EDGE_RELATION_EXPLOITS, weight=0.85)

            # Degrade DEPENDS_ON service edge weights
            for u, v, data in graph.edges(data=True):
                if data.get("relation") == EDGE_RELATION_DEPENDS_ON:
                    if target_sat == v:
                        graph[u][v]["weight"] = round(graph[u][v]["weight"] * 0.15, 3)

            graph.nodes[target_sat]["status"] = "DEGRADED"
            logger.debug("[CTIG Evolution] Injected RESOURCE_EXHAUSTION threat indicator on %s", target_sat)

        # ---------------------------------------------------------------------
        # Compute Health Classification Outputs
        # ---------------------------------------------------------------------
        compromised = [
            n for n, d in graph.nodes(data=True)
            if d.get("status") in ("COMPROMISED", "DEGRADED") or d.get("type") == NODE_TYPE_THREAT
        ]
        healthy = [
            n for n, d in graph.nodes(data=True)
            if d.get("status") == "HEALTHY" and d.get("type") != NODE_TYPE_THREAT
        ]

        # -----------------------------------------------------------------
        # MODULE 4: Compute PageRank Centrality for SPOF analysis
        # Identifies which nodes are most critical to overall graph connectivity.
        # High-centrality compromised nodes represent severe single-points-of-failure.
        # -----------------------------------------------------------------
        centrality_scores = self._compute_pagerank_centrality(graph)

        # -----------------------------------------------------------------
        # MODULE 4: Dijkstra Attack Path Analysis — Critical Corridors
        # For each active THREAT node, compute shortest reachable paths to
        # all critical SERVICE and SATELLITE nodes. These paths represent
        # vulnerable routing corridors that Stage 5 (MIA) and Stage 6 (OPT)
        # can use to block or re-route around compromised routes.
        # -----------------------------------------------------------------
        critical_corridors = []
        threat_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("type") == NODE_TYPE_THREAT
        ]
        for threat_node in threat_nodes:
            paths = self._analyze_attack_paths(graph, threat_node)
            for path in paths:
                # Compute total path weight (sum of inverse edge weights = attack cost)
                path_weight = 0.0
                for i in range(len(path) - 1):
                    edge_data = graph.get_edge_data(path[i], path[i + 1], default={})
                    path_weight += 1.0 / max(0.01, edge_data.get("weight", 1.0))
                critical_corridors.append({
                    "threat_source": threat_node,
                    "target": path[-1] if path else "",
                    "path": path,
                    "path_weight": round(path_weight, 4),
                })
        if critical_corridors:
            logger.info("[CTIG] Identified %d critical attack corridors via Dijkstra analysis.", len(critical_corridors))

        return CTIGOutput(
            graph=graph,
            compromised_nodes=compromised,
            healthy_nodes=healthy,
            num_edges=graph.number_of_edges(),
            centrality_scores=centrality_scores,
            critical_corridors=critical_corridors,
        )

    # =========================================================================
    # Hollow Scaffolding & Advanced Relational Algorithm # TODO Placeholders
    # =========================================================================

    def _compute_pagerank_centrality(self, graph: nx.DiGraph) -> Dict[str, float]:
        """
        # TODO: Advanced Graph Relational Algorithm Slot - Node Centrality / PageRank
        Calculates NetworkX PageRank centrality metrics over the dynamic constellation topology
        to identify high-risk single points of failure under active attack scenarios.
        """
        try:
            return nx.pagerank(graph, weight="weight")
        except Exception as err:
            logger.warning("[CTIG] PageRank computation fallback: %s", err)
            return {node: 1.0 / graph.number_of_nodes() for node in graph.nodes()}

    def _extract_gnn_features(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        # TODO: Advanced Graph Neural Network (GNN) Feature Extractor Slot
        Converts the NetworkX DiGraph into tensor node feature matrices (X) and adjacency
        matrices (A / edge_index) suitable for PyTorch Geometric (PyG) GNN inference.
        """
        # Placeholder for PyTorch Geometric tensor conversion
        node_list = list(graph.nodes())
        node_indices = {node: i for i, node in enumerate(node_list)}
        edge_index = [[node_indices[u], node_indices[v]] for u, v in graph.edges()]

        return {
            "num_nodes": len(node_list),
            "num_edges": len(edge_index),
            "edge_index": edge_index,
            "node_mapping": node_indices,
            "# TODO": "Swap with torch_geometric.utils.from_networkx when GNN pipeline is activated.",
        }

    def _analyze_attack_paths(self, graph: nx.DiGraph, source_threat: str) -> List[List[str]]:
        """
        # TODO: Shortest-Path Attack Propagation Slot (Dijkstra / Bellman-Ford)
        Computes all reachable downstream target paths from a newly injected THREAT_INDICATOR
        node to critical payload and navigation services.
        """
        if source_threat not in graph:
            return []

        paths = []
        for node in graph.nodes():
            if graph.nodes[node].get("type") in (NODE_TYPE_SERVICE, NODE_TYPE_SATELLITE):
                if nx.has_path(graph, source_threat, node):
                    p = nx.shortest_path(graph, source=source_threat, target=node)
                    paths.append(p)
        return paths

    # =========================================================================
    # Stage Execution Contracts & Interface Wrappers
    # =========================================================================

    def execute(self, input_data: Union[StateVectorData, StateVector, TelemetryData, Dict[str, Any]]) -> CTIGOutput:
        """
        Executes Stage 2 graph construction and dynamic threat evolution using standardized interfaces.
        
        Args:
            input_data: Standardized StateVectorData, StateVector object, TelemetryData, or composite dict.
            
        Returns:
            CTIGOutput: Standardized Threat Intelligence Graph artifact containing NetworkX DiGraph.
        """
        raw_telemetry: Optional[TelemetryData] = None

        if isinstance(input_data, StateVectorData):
            s_t = input_data
        elif isinstance(input_data, StateVector):
            s_t = StateVectorData(
                C=input_data.s_t.get("C", 1.0),
                R=input_data.s_t.get("R", 1.0),
                T=input_data.s_t.get("T", 1.0),
                Q=input_data.s_t.get("Q", 1.0),
                M=input_data.s_t.get("M", 1.0),
                E=input_data.s_t.get("E", 1.0),
                A=input_data.s_t.get("A", 1.0),
                timestamp=input_data.timestamp,
                active_threats=input_data.active_threats,
            )
        elif isinstance(input_data, TelemetryData):
            raw_telemetry = input_data
            s_t = StateVectorData(timestamp=input_data.timestamp)
        elif isinstance(input_data, dict):
            if "state_vector" in input_data and isinstance(input_data["state_vector"], StateVectorData):
                s_t = input_data["state_vector"]
                if "telemetry" in input_data and isinstance(input_data["telemetry"], TelemetryData):
                    raw_telemetry = input_data["telemetry"]
            else:
                raw_telemetry = TelemetryData.from_dict(input_data)
                s_t = StateVectorData(timestamp=raw_telemetry.timestamp)
        else:
            s_t = StateVectorData()

        logger.info("[Stage 2 CTIG] Building dynamic threat graph for %d active threat flags.", len(s_t.active_threats))
        return self._graph_builder_heuristic_slot(s_t, raw_telemetry)

    def process(self, state: Any) -> Dict[str, Any]:
        """Backward compatibility bridge returning dictionary representation."""
        output = self.execute(state)
        return {
            "num_nodes": output.graph.number_of_nodes(),
            "num_edges": output.num_edges,
            "compromised_nodes": output.compromised_nodes,
            "healthy_nodes": output.healthy_nodes,
        }


# Backward compatibility alias
Stage2CTIG = CognitiveThreatIntelligenceGraph


# =============================================================================
# Standalone Verification Suite (Unit Tests for Nominal, Jamming, Spoofing, DoS, Healing)
# =============================================================================
if __name__ == "__main__":
    from src.stages.stage1_csa import CyberSituationAwareness

    print("==========================================================================")
    print("      HCCRO Stage 2: Cognitive Threat Intelligence Graph (CTIG) Test      ")
    print("==========================================================================")

    csa = CyberSituationAwareness()
    ctig = CognitiveThreatIntelligenceGraph()

    # 1. NOMINAL SCENARIO
    print("\n--- [Scenario 1: Nominal Operations] ---")
    telemetry_nominal = {
        "packet_delivery_ratio": 0.99,
        "communication_latency": 20.0,
        "spectrum_usage": 0.05,
        "navigation_consistency": 1.2,
        "authentication_events": 0,
        "processor_utilization": 15.0,
        "memory_consumption": 22.0,
    }
    s_t_nominal = csa.execute(telemetry_nominal)
    output_nominal = ctig.execute(s_t_nominal)
    g_nom = output_nominal.graph

    print(f"Active Vertices ({g_nom.number_of_nodes()} total): {list(g_nom.nodes())}")
    print(f"Active Edges ({output_nominal.num_edges} total): {list(g_nom.edges())[:5]}... (sample)")
    print(f"Compromised Nodes: {output_nominal.compromised_nodes}")
    print(f"Healthy Nodes Count: {len(output_nominal.healthy_nodes)}")
    assert len(output_nominal.compromised_nodes) == 0, "Nominal graph should have zero compromised nodes!"
    print("-> Verification Passed: Nominal Graph Built Cleanly.")

    # 2. RF JAMMING SCENARIO
    print("\n--- [Scenario 2: RF Jamming Anomaly] ---")
    telemetry_jamming = {
        "packet_delivery_ratio": 0.20,
        "communication_latency": 450.0,
        "spectrum_usage": 0.85,
        "processor_utilization": 20.0,
    }
    s_t_jamming = csa.execute(telemetry_jamming)
    output_jamming = ctig.execute(s_t_jamming)
    g_jam = output_jamming.graph

    print(f"Active Vertices ({g_jam.number_of_nodes()} total): {list(g_jam.nodes())}")
    print("Active Threat Nodes:", [n for n, d in g_jam.nodes(data=True) if d.get("type") == NODE_TYPE_THREAT])
    exploits_edges = [(u, v, d) for u, v, d in g_jam.edges(data=True) if d.get("relation") == EDGE_RELATION_EXPLOITS]
    comm_edges = [(u, v, d["weight"]) for u, v, d in g_jam.edges(data=True) if d.get("relation") == EDGE_RELATION_COMMUNICATES_WITH and "SAT_LEO_01" in (u, v)]
    print(f"Injected EXPLOITS Edges: {exploits_edges}")
    print(f"Degraded COMMUNICATES_WITH Edge Weights: {comm_edges[:3]}")
    print(f"Compromised Nodes: {output_jamming.compromised_nodes}")
    assert "THREAT_RF_JAMMING_SAT_LEO_01" in g_jam, "RF Jamming threat node missing!"
    print("-> Verification Passed: RF Jamming Anomaly Correctly Graph-Mapped.")

    # 3. GNSS SPOOFING SCENARIO
    print("\n--- [Scenario 3: GNSS Spoofing Anomaly] ---")
    telemetry_spoofing = {
        "pseudorange_error_m": 45.0,
        "authentication_events": 5,
        "packet_delivery_ratio": 0.95,
    }
    s_t_spoofing = csa.execute(telemetry_spoofing)
    output_spoofing = ctig.execute(s_t_spoofing)
    g_spf = output_spoofing.graph

    print(f"Active Vertices ({g_spf.number_of_nodes()} total): {list(g_spf.nodes())}")
    print("Active Threat Nodes:", [n for n, d in g_spf.nodes(data=True) if d.get("type") == NODE_TYPE_THREAT])
    trust_edges = [(u, v, d["weight"]) for u, v, d in g_spf.edges(data=True) if d.get("relation") == EDGE_RELATION_TRUSTS]
    print(f"Degraded TRUSTS Edge Weights: {trust_edges[:4]}")
    print(f"Compromised Nodes: {output_spoofing.compromised_nodes}")
    assert "THREAT_GNSS_SPOOFING_SAT_LEO_01" in g_spf, "GNSS Spoofing threat node missing!"
    print("-> Verification Passed: GNSS Spoofing Anomaly Correctly Graph-Mapped.")

    # 4. RESOURCE EXHAUSTION (DoS) SCENARIO
    print("\n--- [Scenario 4: Resource Exhaustion DoS Anomaly] ---")
    telemetry_dos = {
        "processor_utilization": 98.0,
        "memory_consumption": 95.0,
        "packet_delivery_ratio": 0.95,
    }
    s_t_dos = csa.execute(telemetry_dos)
    output_dos = ctig.execute(s_t_dos)
    g_dos = output_dos.graph

    print(f"Active Vertices ({g_dos.number_of_nodes()} total): {list(g_dos.nodes())}")
    print("Active Threat Nodes:", [n for n, d in g_dos.nodes(data=True) if d.get("type") == NODE_TYPE_THREAT])
    depends_edges = [(u, v, d["weight"]) for u, v, d in g_dos.edges(data=True) if d.get("relation") == EDGE_RELATION_DEPENDS_ON and v == "SAT_LEO_01"]
    print(f"Degraded DEPENDS_ON Edge Weights: {depends_edges}")
    print(f"Compromised Nodes: {output_dos.compromised_nodes}")
    assert "THREAT_RESOURCE_EXHAUSTION_SAT_LEO_01" in g_dos, "Resource Exhaustion threat node missing!"
    print("-> Verification Passed: Resource Exhaustion Anomaly Correctly Graph-Mapped.")

    # 5. GRAPH HEALING RECOVERY SCENARIO
    print("\n--- [Scenario 5: Graph Healing & Recovery Verification] ---")
    print("Simulating return to nominal telemetry after attack mitigation...")
    s_t_healed = csa.execute(telemetry_nominal)
    output_healed = ctig.execute(s_t_healed)
    g_healed = output_healed.graph

    threat_nodes_remaining = [n for n, d in g_healed.nodes(data=True) if d.get("type") == NODE_TYPE_THREAT]
    print(f"Remaining Threat Indicator Nodes: {threat_nodes_remaining}")
    print(f"Compromised Nodes Count: {len(output_healed.compromised_nodes)}")
    print(f"Healthy Nodes Count: {len(output_healed.healthy_nodes)}")

    comm_healed_weights = [d["weight"] for u, v, d in g_healed.edges(data=True) if d.get("relation") == EDGE_RELATION_COMMUNICATES_WITH]
    assert len(threat_nodes_remaining) == 0, "Threat indicator nodes were not healed/cleared!"
    assert len(output_healed.compromised_nodes) == 0, "Compromised node list not empty after healing!"
    assert all(w == 1.0 for w in comm_healed_weights), "Communication edge weights were not restored to 1.0!"
    print("-> Verification SUCCESS: Graph fully healed and restored to nominal operational state!")
    print("\n==========================================================================")
