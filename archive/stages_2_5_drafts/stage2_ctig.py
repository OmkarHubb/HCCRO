"""
src/stages/stage2_ctig.py

Stage 2 -- Cognitive Threat Intelligence Graph (CTIG)

Builds a dynamic networkx.DiGraph representing:
  - `satellite_node` vertices (physical satellites)
  - `hardware_service` vertices (each satellite's transceiver, navigation,
    and compute/routing services)
  - `communication_link` edges between satellites (inter-satellite P2P
    links, carrying a `weight` in [0,1] representing current link
    trust/quality)
  - `depends_on` edges from a satellite to its own onboard services

and dynamically injects THREAT_INDICATOR nodes -- connected via directed
`exploits` edges to the specific *named* hardware_service node they target
(`navigation_service` under spoofing, `transceiver_service` under jamming,
`compute_service` under DoS) -- whenever `update_ctig()` is called with an
anomalous telemetry reading. Injection also degrades the `weight` of every
`communication_link` edge touching the compromised satellite, so Stage 4's
propagation model sees a genuinely weakened pathway rather than a purely
cosmetic graph annotation.

Default topology: a 5-satellite cycle graph SC1 -> SC2 -> ... -> SC5 -> SC1,
per the simulated cluster spec (`build_cycle_topology`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import networkx as nx

from src.utils.state_vector import StateVector, coerce_state_vector

# ---------------------------------------------------------------------------
# Anomaly thresholds (spec sections 4.1 / 4.2)
# ---------------------------------------------------------------------------
CN0_JAMMING_THRESHOLD_DB_HZ = 25.0     # C/N0 below this => RF jamming
GPS_DRIFT_SPOOFING_THRESHOLD_M = 15.0  # combined lat/lon/alt drift => spoofing
CPU_DOS_THRESHOLD_PCT = 90.0           # CPU/buffer utilization spike => DoS

# How much a single threat injection degrades an incident communication_link
# edge's weight, scaled by severity (weight is clamped to a floor so it
# never hits exactly zero / disconnects the graph outright).
LINK_DEGRADATION_FACTOR = 0.5
LINK_WEIGHT_FLOOR = 0.05

DEFAULT_SATELLITE_IDS = [f"SC{i}" for i in range(1, 6)]


@dataclass
class ThreatEvent:
    threat_node_id: str
    subtype: str            # "JAMMING" | "SPOOFING" | "DoS"
    satellite_id: str
    targeted_nodes: List[str]
    severity: float          # 0-1
    degraded_links: List[tuple]  # [(u, v), ...] edges whose weight was cut


class CognitiveThreatGraph:
    """Maintains the evolving Cognitive Threat Intelligence Graph (Stage 2)."""

    NODE_TYPE_SATELLITE = "satellite_node"
    NODE_TYPE_SERVICE = "hardware_service"
    NODE_TYPE_THREAT = "THREAT_INDICATOR"

    EDGE_DEPENDS_ON = "depends_on"
    EDGE_COMMUNICATION_LINK = "communication_link"
    EDGE_EXPLOITS = "exploits"
    EDGE_TARGETS = "targets"

    # Anomaly subtype -> the specific named hardware_service it targets
    SUBTYPE_TARGET_SERVICE = {
        "JAMMING": "transceiver_service",
        "SPOOFING": "navigation_service",
        "DoS": "compute_service",
    }

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._threat_seq = itertools.count(1)

    # ------------------------------------------------------------------
    # Topology construction
    # ------------------------------------------------------------------
    @classmethod
    def build_cycle_topology(cls, satellite_ids: Optional[List[str]] = None) -> "CognitiveThreatGraph":
        """Convenience constructor for the standard 5-satellite cycle graph
        SC1 -> SC2 -> SC3 -> SC4 -> SC5 -> SC1 used by the simulated cluster."""
        ids = satellite_ids or DEFAULT_SATELLITE_IDS
        ctig = cls()
        for sid in ids:
            ctig.add_satellite(sid)
        for a, b in zip(ids, ids[1:] + ids[:1]):
            ctig.link_satellites(a, b, weight=1.0)
        return ctig

    def add_satellite(self, satellite_id: str, neighbors: Optional[Iterable[str]] = None) -> None:
        """Register a satellite plus its three onboard hardware_service
        nodes (transceiver, navigation, compute), and its communication
        links to already-known neighbors."""

        transceiver_id = f"{satellite_id}::transceiver_service"
        navigation_id = f"{satellite_id}::navigation_service"
        compute_id = f"{satellite_id}::compute_service"

        self.graph.add_node(satellite_id, node_type=self.NODE_TYPE_SATELLITE)
        for service_id, service_kind in (
            (transceiver_id, "transceiver_service"),
            (navigation_id, "navigation_service"),
            (compute_id, "compute_service"),
        ):
            self.graph.add_node(service_id, node_type=self.NODE_TYPE_SERVICE,
                                 service_kind=service_kind, satellite_id=satellite_id)
            self.graph.add_edge(satellite_id, service_id, relation=self.EDGE_DEPENDS_ON)

        for neighbor_id in neighbors or []:
            if self.graph.has_node(neighbor_id):
                self.link_satellites(satellite_id, neighbor_id)

    def link_satellites(self, sat_a: str, sat_b: str, weight: float = 1.0) -> None:
        """Bidirectional `communication_link` edges carrying a `weight` in
        [0,1] -- current link trust/quality. Stage 4's ICM reads this weight
        directly as the inverse propagation probability, so degrading it
        here (via `update_ctig`) has a real downstream effect."""
        self.graph.add_edge(sat_a, sat_b, relation=self.EDGE_COMMUNICATION_LINK, weight=weight)
        self.graph.add_edge(sat_b, sat_a, relation=self.EDGE_COMMUNICATION_LINK, weight=weight)

    # ------------------------------------------------------------------
    # Dynamic anomaly ingestion -> THREAT_INDICATOR injection
    # ------------------------------------------------------------------
    def update_ctig(self, satellite_id: str, telemetry: dict,
                     state: Union[StateVector, dict]) -> List[ThreatEvent]:
        """Evaluate one tick of raw telemetry for `satellite_id` and inject
        any THREAT_INDICATOR nodes the anomaly thresholds warrant, degrading
        that satellite's communication_link edges in proportion to severity.
        Returns the list of events injected this call (possibly empty).

        `telemetry` is a plain dict of raw readings, e.g.:
            {"cn0_db_hz": 12.0, "lat_drift_m": 0.0, "lon_drift_m": 0.0,
             "alt_error_m": 0.0, "cpu_util_pct": 30.0, "buffer_fill_pct": 20.0}
        Any keys not supplied default to nominal values.
        `state` may be a StateVector or a plain dict {"C":..., ..., "A":...}
        (Input Contract) -- it is not used for thresholding (that's driven
        by raw telemetry) but is attached to injected threat nodes for
        downstream context.
        """
        state = coerce_state_vector(state)

        if not self.graph.has_node(satellite_id):
            self.add_satellite(satellite_id)

        cn0 = telemetry.get("cn0_db_hz", 45.0)
        lat_drift = telemetry.get("lat_drift_m", 0.0)
        lon_drift = telemetry.get("lon_drift_m", 0.0)
        alt_error = telemetry.get("alt_error_m", 0.0)
        cpu_util = telemetry.get("cpu_util_pct", 20.0)
        buffer_fill = telemetry.get("buffer_fill_pct", 20.0)

        events: List[ThreatEvent] = []

        # --- JAMMING: C/N0 collapse (Mendeley GNSS mapping, section 4.1) ---
        if cn0 < CN0_JAMMING_THRESHOLD_DB_HZ:
            severity = min(1.0, (CN0_JAMMING_THRESHOLD_DB_HZ - cn0) / CN0_JAMMING_THRESHOLD_DB_HZ)
            events.append(self._inject_threat(satellite_id, "JAMMING", severity))

        # --- SPOOFING: GPS coordinate drift (Mendeley GNSS mapping) --------
        drift = abs(lat_drift) + abs(lon_drift) + abs(alt_error)
        if drift > GPS_DRIFT_SPOOFING_THRESHOLD_M:
            severity = min(1.0, drift / (GPS_DRIFT_SPOOFING_THRESHOLD_M * 4))
            events.append(self._inject_threat(satellite_id, "SPOOFING", severity))

        # --- DoS: host resource-saturation curve (simulated, see spec §2) --
        if cpu_util > CPU_DOS_THRESHOLD_PCT or buffer_fill > CPU_DOS_THRESHOLD_PCT:
            severity = min(1.0, max(cpu_util, buffer_fill) / 100.0)
            events.append(self._inject_threat(satellite_id, "DoS", severity))

        return events

    # kept for backward compatibility with earlier callers
    def ingest(self, satellite_id: str, telemetry: dict, state) -> List[ThreatEvent]:
        return self.update_ctig(satellite_id, telemetry, state)

    def _inject_threat(self, satellite_id: str, subtype: str, severity: float) -> ThreatEvent:
        threat_id = f"THREAT::{subtype}::{satellite_id}::{next(self._threat_seq)}"
        target_service = f"{satellite_id}::{self.SUBTYPE_TARGET_SERVICE[subtype]}"

        self.graph.add_node(
            threat_id,
            node_type=self.NODE_TYPE_THREAT,
            subtype=subtype,
            origin_satellite=satellite_id,
            severity=severity,
        )
        if self.graph.has_node(target_service):
            self.graph.add_edge(threat_id, target_service, relation=self.EDGE_EXPLOITS, severity=severity)
        self.graph.add_edge(threat_id, satellite_id, relation=self.EDGE_TARGETS, severity=severity)

        degraded_links = self._degrade_incident_links(satellite_id, severity)

        return ThreatEvent(threat_node_id=threat_id, subtype=subtype,
                            satellite_id=satellite_id, targeted_nodes=[target_service],
                            severity=severity, degraded_links=degraded_links)

    def _degrade_incident_links(self, satellite_id: str, severity: float) -> List[tuple]:
        """Cuts the `weight` of every communication_link edge touching
        `satellite_id`, proportional to the injected threat's severity, to
        simulate active pathway degradation."""
        degraded = []
        for u, v, data in list(self.graph.edges(data=True)):
            if data.get("relation") != self.EDGE_COMMUNICATION_LINK:
                continue
            if satellite_id not in (u, v):
                continue
            current = data.get("weight", 1.0)
            new_weight = max(LINK_WEIGHT_FLOOR, current * (1.0 - LINK_DEGRADATION_FACTOR * severity))
            self.graph[u][v]["weight"] = new_weight
            degraded.append((u, v))
        return degraded

    # ------------------------------------------------------------------
    # Helpers for downstream stages
    # ------------------------------------------------------------------
    def active_threats(self) -> List[str]:
        return [n for n, d in self.graph.nodes(data=True)
                if d.get("node_type") == self.NODE_TYPE_THREAT]

    def satellites(self) -> List[str]:
        return [n for n, d in self.graph.nodes(data=True)
                if d.get("node_type") == self.NODE_TYPE_SATELLITE]

    def communication_links(self) -> List[tuple]:
        return [(u, v, d) for u, v, d in self.graph.edges(data=True)
                if d.get("relation") == self.EDGE_COMMUNICATION_LINK]

    def prune_threat(self, threat_node_id: str) -> None:
        """Remove a resolved threat indicator (call this from Stage 7 after
        a successful self-healing action). Does not restore link weights --
        that recovery is Stage 7's job."""
        if self.graph.has_node(threat_node_id):
            self.graph.remove_node(threat_node_id)
