"""
Module 1: Layer 1 — Satellite Node Agent (SatelliteNode)
=========================================================
Represents an individual spacecraft in the constellation. Executes the local
8-stage HCCRO cognitive loop, tracks its local 7-D State Vector S_t, and manages
local actuators, resource budgets, and execution history.
"""

from typing import Dict, Any, List, Optional
from src.core.interfaces import StateVectorData, PaceState
from src.core.orchestrator import HCCROOrchestrator
from src.utils.logger import get_logger

logger = get_logger("Hierarchy.SatelliteNode")


class SatelliteNode:
    """
    Layer 1 Agent: Individual Satellite Node.
    """

    def __init__(self, node_id: str, cluster_id: str = "CLUSTER_01"):
        self.node_id = node_id
        self.cluster_id = cluster_id
        self.status = "HEALTHY"  # HEALTHY, DEGRADED, COMPROMISED, ISOLATED
        self.orchestrator = HCCROOrchestrator()
        self.state_vector = StateVectorData()
        self.execution_history: List[Dict[str, Any]] = []

    def step(self, telemetry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes one local 8-stage HCCRO cognitive cycle for this satellite.
        
        Args:
            telemetry: Optional raw telemetry override dict or TelemetryData.
            
        Returns:
            Dict containing full stage outputs, state vector, and optimal actions.
        """
        logger.info("[%s] Executing local 8-stage HCCRO cognitive cycle...", self.node_id)
        
        # Run local 8-stage pipeline
        result = self.orchestrator.run_pipeline(telemetry_data=telemetry)
        
        # Update local state vector from pipeline output
        if "state_vector" in result and isinstance(result["state_vector"], StateVectorData):
            self.state_vector = result["state_vector"]
        elif "stage1_state_vector" in result and hasattr(result["stage1_state_vector"], "state_vector"):
            self.state_vector = result["stage1_state_vector"].state_vector

        # Update node health status based on active threats & resilience score R
        if "GPS_SPOOFING_SUSPECTED" in self.state_vector.active_threats or "RF_JAMMING_SUSPECTED" in self.state_vector.active_threats:
            if self.state_vector.R < 0.3:
                self.status = "COMPROMISED"
            else:
                self.status = "DEGRADED"
        elif self.status != "ISOLATED":
            self.status = "HEALTHY"

        history_entry = {
            "node_id": self.node_id,
            "status": self.status,
            "pace_state": self.state_vector.pace_state.value,
            "resilience_R": self.state_vector.R,
            "trust_T": self.state_vector.T,
            "energy_E": self.state_vector.E,
            "active_threats": list(self.state_vector.active_threats),
            "optimal_actions": (
                result.get("mitigation_plan").optimal_actions if hasattr(result.get("mitigation_plan"), "optimal_actions")
                else result.get("stage6_optimization", {}).get("optimal_actions", [])
            ),
        }
        self.execution_history.append(history_entry)
        
        result["node_id"] = self.node_id
        result["cluster_id"] = self.cluster_id
        result["node_status"] = self.status
        return result

    def get_telemetry_digest(self) -> Dict[str, Any]:
        """Returns node telemetry and state vector summary for cluster aggregation."""
        return {
            "node_id": self.node_id,
            "cluster_id": self.cluster_id,
            "status": self.status,
            "pace_state": self.state_vector.pace_state.value,
            "state_vector": self.state_vector.to_dict(),
            "active_threats": list(self.state_vector.active_threats),
            "cpu_load": round(1.0 - self.state_vector.E, 4),
        }

    def apply_actuator(self, action: str) -> bool:
        """Executes a local countermeasure actuator on this spacecraft."""
        logger.info("[%s] Applying local actuator: %s", self.node_id, action)
        if action in ["ACTIVATE_FREQUENCY_HOPPING", "TRIGGER_FREQUENCY_HOPPING"]:
            self.state_vector.C = min(1.0, round(self.state_vector.C + 0.25, 4))
            self.state_vector.Q = min(1.0, round(self.state_vector.Q + 0.20, 4))
            return True
        elif action in ["SWITCH_TO_INERTIAL_NAV_FALLBACK", "REVERT_TO_IMU_NAV"]:
            self.state_vector.T = min(1.0, round(self.state_vector.T + 0.30, 4))
            return True
        elif action in ["ENFORCE_PROCESS_QUOTA_ISOLATION", "KILL_DoS_PROCESS"]:
            self.state_vector.E = min(1.0, round(self.state_vector.E + 0.35, 4))
            return True
        elif action == "ISOLATE_NODE":
            self.isolate()
            return True
        return False

    def isolate(self) -> None:
        """Isolates satellite node from cluster network communications."""
        self.status = "ISOLATED"
        self.state_vector.C = 0.0
        self.state_vector.pace_state = PaceState.EMERGENCY
        logger.warning("[%s] Satellite NODE ISOLATED due to severe security compromise!", self.node_id)
