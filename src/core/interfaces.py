"""
HCCRO Core Interfaces & Data Contracts
======================================
Defines standardized, strongly-typed data objects, Enum types, and abstract stage contracts
grounded in Cognitive Cyber Resilience Theory (CCRT) and PACE Edge Fallback Architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
import time
import networkx as nx


class PaceState(str, Enum):
    """
    Onboard Physical PACE Fallback Sub-layer States.
    """
    PRIMARY = "PRIMARY"            # Nominal operational state
    ALTERNATE = "ALTERNATE"        # Degraded state (minor threat/resource pressure)
    CONTINGENCY = "CONTINGENCY"    # Survivability state (severe threat/depletion)
    EMERGENCY = "EMERGENCY"        # Minimal-operation safe state (low power recharge)


@dataclass
class TelemetryData:
    """
    Standardized input interface for raw satellite telemetry data feeds.
    Supports the 13 core CCRT / PACE metrics with dynamic extra parameters.
    """
    packet_delivery_ratio: float = 1.0            # PDR (0.0 to 1.0)
    communication_latency: float = 0.0            # Latency in ms
    spectrum_usage: float = 0.0                   # Spectrum noise ratio (0.0 to 1.0)
    navigation_consistency: float = 0.0           # GPS pseudorange error/drift in meters
    authentication_events: int = 0                # Number of failed auth events
    processor_utilization: float = 0.0            # CPU usage percentage (0.0 to 100.0)
    memory_consumption: float = 0.0               # RAM usage percentage (0.0 to 100.0)
    energy_availability: float = 1.0              # Battery level / charge fraction (0.0 to 1.0)
    routing_status: float = 1.0                   # Routing stability metric (0.0 to 1.0)
    software_integrity_measurements: float = 1.0  # Software integrity index (0.0 to 1.0)
    trust_values: float = 1.0                     # Trust score metric (0.0 to 1.0)
    telemetry_status: float = 1.0                 # Telemetry link status (0.0 to 1.0)
    environmental_conditions: float = 0.0         # Environmental radiation/space weather index
    timestamp: float = field(default_factory=time.time)
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryData":
        """Factory method to parse a raw dictionary into TelemetryData with aliasing and fallbacks."""
        known_keys = {
            "packet_delivery_ratio", "communication_latency", "spectrum_usage",
            "navigation_consistency", "authentication_events",
            "processor_utilization", "memory_consumption", "energy_availability",
            "routing_status", "software_integrity_measurements", "trust_values",
            "telemetry_status", "environmental_conditions", "timestamp",
            "snr_db", "pseudorange_error_m", "cpu_usage_pct", "memory_usage_pct", "power_level_pct"
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        # Aliased metric conversions
        if "packet_delivery_ratio" in data:
            pdr = float(data["packet_delivery_ratio"])
        elif "snr_db" in data:
            pdr = float(max(0.0, min(1.0, data["snr_db"] / 25.0)))
        else:
            pdr = 1.0

        if "spectrum_usage" in data:
            spectrum = float(data["spectrum_usage"])
        elif "snr_db" in data:
            spectrum = float(max(0.0, min(1.0, (30.0 - data["snr_db"]) / 30.0)))
        else:
            spectrum = 0.0

        nav_drift = float(data.get("navigation_consistency", data.get("pseudorange_error_m", 0.0)))
        auth_events = int(data.get("authentication_events", 4 if nav_drift > 15.0 else 0))
        cpu = float(data.get("processor_utilization", data.get("cpu_usage_pct", 0.0)))
        ram = float(data.get("memory_consumption", data.get("memory_usage_pct", 0.0)))

        if "energy_availability" in data:
            energy = float(data["energy_availability"])
        elif "power_level_pct" in data:
            energy = float(data["power_level_pct"]) / 100.0
        else:
            energy = 1.0

        latency = float(data.get("communication_latency", 400.0 if (pdr < 0.5 or cpu > 80.0) else 0.0))
        routing = float(data.get("routing_status", 1.0 / (1.0 + (latency / 100.0))))
        sw_integrity = float(data.get("software_integrity_measurements", 1.0))
        trust_val = float(data.get("trust_values", 1.0))
        tlm_stat = float(data.get("telemetry_status", 1.0))
        env_cond = float(data.get("environmental_conditions", 0.0))

        return cls(
            packet_delivery_ratio=pdr,
            communication_latency=latency,
            spectrum_usage=spectrum,
            navigation_consistency=nav_drift,
            authentication_events=auth_events,
            processor_utilization=cpu,
            memory_consumption=ram,
            energy_availability=energy,
            routing_status=routing,
            software_integrity_measurements=sw_integrity,
            trust_values=trust_val,
            telemetry_status=tlm_stat,
            environmental_conditions=env_cond,
            timestamp=float(data.get("timestamp", time.time())),
            extra_metrics=extra,
        )


@dataclass
class StateVectorData:
    """
    Standardized Operational State Vector S_t = {C, R, T, Q, M, E, A}.
    Every dimension is normalized and strictly bounded in [0.0, 1.0].
    """
    C: float = 1.0  # Communication Integrity
    R: float = 1.0  # Routing Stability
    T: float = 1.0  # Trust State
    Q: float = 1.0  # Communication Quality
    M: float = 1.0  # Mission Performance
    E: float = 1.0  # Resource Efficiency
    A: float = 1.0  # Autonomous Decision Reliability
    pace_state: PaceState = PaceState.PRIMARY
    timestamp: float = field(default_factory=time.time)
    active_threats: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        """Serializes state metrics to standard dictionary."""
        return {
            "C": self.C,
            "R": self.R,
            "T": self.T,
            "Q": self.Q,
            "M": self.M,
            "E": self.E,
            "A": self.A,
        }


@dataclass
class CTIGOutput:
    """Stage 2: Cognitive Threat Intelligence Graph Output Artifact."""
    graph: nx.DiGraph
    compromised_nodes: List[str]
    healthy_nodes: List[str]
    num_edges: int
    # MODULE 4: PageRank centrality scores keyed by node id
    centrality_scores: Dict[str, float] = field(default_factory=dict)
    # MODULE 4: Vulnerable routing corridors identified by Dijkstra shortest-path
    # Each entry: {"threat_source": str, "path": List[str], "path_weight": float}
    critical_corridors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AIMOutput:
    """Stage 3: Attack Intention Modeling Output Artifact."""
    intention_scores: Dict[str, float]
    primary_intention: str
    confidence_level: float


@dataclass
class AEPOutput:
    """Stage 4: Attack Evolution Prediction Output Artifact."""
    predicted_paths: List[Dict[str, Any]]
    lateral_propagation_probabilities: Dict[str, float]
    escalation_risk: float


@dataclass
class MIAOutput:
    """Stage 5: Mission Impact Estimation Output Artifact."""
    mci_score: float
    cri_score: float
    mdi_score: float
    threat_severity: str
    affected_subsystems: List[str]
    threat_adjusted_utilities: Dict[str, float]


@dataclass
class OptimizationOutput:
    """Stage 6: Multi-Objective Resilience Optimization Output Artifact."""
    optimal_actions: List[str]
    target_pace_state: PaceState
    utility_score: float
    drei_score: float
    weights_applied: Dict[str, float]
    # MODULE 2: Resource allocation decision vector from constrained SLSQP solver
    # Keys: cpu_security, cpu_payload, power_security, power_payload, bw_security, bw_payload
    resource_allocation: Dict[str, float] = field(default_factory=dict)
    # MODULE 2: Whether constraints were satisfied by the solver
    constraints_satisfied: bool = True


@dataclass
class HealingOutput:
    """Stage 7: Distributed Self-Healing Execution Output Artifact."""
    status: str
    current_pace_state: PaceState
    executed_actions: List[str]
    execution_timeline_ms: float
    action_confirmations: Dict[str, bool]


@dataclass
class CKEOutput:
    """Stage 8: Cyber Knowledge Evolution Output Artifact."""
    record_id: str
    logged_at: float
    persisted_successfully: bool
    historical_success_rate: float


class BaseStage(ABC):
    """
    Abstract Base Class for all HCCRO pipeline stages.
    Enforces a standard execution interface for modular plug-and-play capability.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """Executes the stage logic using standard input/output contracts."""
        pass
