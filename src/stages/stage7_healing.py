"""
Stage 7: Distributed Self-Healing (The Physical Actuators)
==========================================================
Executes selected countermeasure actions using a SelfHealingActuator callback registry,
applies physical state vector recovery mathematical shifts, and updates physical PACE fallback sub-layer states.
"""

import time
from typing import Dict, Any, Callable, List, Optional, Union
from src.core.interfaces import BaseStage, OptimizationOutput, HealingOutput, PaceState, StateVectorData
from src.utils.logger import get_logger

logger = get_logger("Stage7.Healing")


class SelfHealingActuator:
    """
    Actuator registry mapping strategic action strings directly to physical execution callbacks
    and applying mathematical state vector recovery shifts.
    """

    def __init__(self):
        self.actuator_registry: Dict[str, Callable[[str], bool]] = {}
        self._register_default_actuators()

    def register_action(self, action_key: str, callback: Callable[[str], bool]) -> None:
        """Registers a custom actuator callback for a strategy key."""
        self.actuator_registry[action_key] = callback
        logger.info("[SelfHealingActuator] Registered actuator callback for '%s'.", action_key)

    def _default_callback(self, action_key: str) -> bool:
        """Simulated default physical hardware actuator execution."""
        logger.info("[Stage 7 Actuator] Physical execution of command: '%s'", action_key)
        return True

    def _register_default_actuators(self) -> None:
        """Registers default actuator callbacks for core resilience actions."""
        actions = [
            "TRIGGER_FREQUENCY_HOPPING",
            "ACTIVATE_FREQUENCY_HOPPING",
            "REVERT_TO_IMU_NAV",
            "SWITCH_TO_INERTIAL_NAV_FALLBACK",
            "KILL_DoS_PROCESS",
            "ENFORCE_PROCESS_QUOTA_ISOLATION",
            "PURGE_MALICIOUS_APID_QUEUE",
            "REVERT_TO_SAFE_TC_KEY_STORE",
            "MIGRATE_TASK",
            "TRANSITION_PACE_EMERGENCY_SAFE_MODE",
            "ISOLATE_COMPROMISED_NODE",
            "RETRIGGER_HANDSHAKE",
            "MAINTAIN_NOMINAL_OPERATIONS",
        ]
        for act in actions:
            self.actuator_registry[act] = self._default_callback

    def execute_action(self, action_key: str) -> bool:
        """Executes registered callback for the action key."""
        callback = self.actuator_registry.get(action_key, self._default_callback)
        try:
            return callback(action_key)
        except Exception as err:
            logger.error("[SelfHealingActuator] Error executing '%s': %s", action_key, err)
            return False

    def heal_state_vector(
        self, s_t: StateVectorData, executed_actions: List[str]
    ) -> StateVectorData:
        """
        Applies physical state recovery mathematical shifts based on executed actions
        and returns a healed StateVectorData artifact.
        """
        # Create a mutable copy of state metrics
        c_val = s_t.C
        r_val = s_t.R
        t_val = s_t.T
        q_val = s_t.Q
        m_val = s_t.M
        e_val = s_t.E
        a_val = s_t.A
        remaining_threats = list(s_t.active_threats)

        for action in executed_actions:
            if action in ("TRIGGER_FREQUENCY_HOPPING", "ACTIVATE_FREQUENCY_HOPPING"):
                # Simulates spectrum reallocation: reset comm quality Q and comm integrity C
                q_val = 1.0
                c_val = 1.0
                r_val = min(1.0, r_val + 0.2)
                remaining_threats = [th for th in remaining_threats if th not in ("RF_JAMMING_SUSPECTED", "RF_JAMMING_DISRUPTION")]
                logger.info("[SelfHealingActuator] Frequency Hopping applied: Restored Q=1.0, C=1.0.")

            elif action in ("REVERT_TO_IMU_NAV", "SWITCH_TO_INERTIAL_NAV_FALLBACK"):
                # Simulates hardware nav bypass: ignore compromised GPS, fallback on IMU, restore Trust T
                t_val = 1.0
                remaining_threats = [th for th in remaining_threats if th not in ("GPS_SPOOFING_SUSPECTED", "GPS_SPOOFING_CORRUPTION")]
                logger.info("[SelfHealingActuator] IMU Navigation Fallback applied: Restored T=1.0, reset drift.")

            elif action in ("KILL_DoS_PROCESS", "ENFORCE_PROCESS_QUOTA_ISOLATION"):
                # Simulates software process reboot: terminate malicious CPU task, restore CPU/Resource Efficiency E
                e_val = 1.0
                m_val = min(1.0, m_val + 0.15)
                remaining_threats = [th for th in remaining_threats if th not in ("RESOURCE_DOS_SUSPECTED", "DOS_ATTACK_ACTIVE")]
                logger.info("[SelfHealingActuator] Process Termination applied: Restored CPU load to nominal 15%, E=1.0.")

            elif action == "MIGRATE_TASK":
                # Simulates payload computing task offload
                e_val = min(1.0, e_val + 0.3)
                m_val = min(1.0, m_val + 0.2)
                logger.info("[SelfHealingActuator] Task Migration applied: Payload computing offloaded to healthy peer.")

            elif action == "PURGE_MALICIOUS_APID_QUEUE":
                # OPSSAT-AD: Clears corrupted telecommand APID queues, restores C and A
                c_val = 1.0
                a_val = 1.0
                remaining_threats = [th for th in remaining_threats if th != "TELECOMMAND_INJECTION_SUSPECTED"]
                logger.info("[SelfHealingActuator] PURGE_MALICIOUS_APID_QUEUE applied: Restored C=1.0, A=1.0.")

            elif action == "REVERT_TO_SAFE_TC_KEY_STORE":
                # OPSSAT-AD: Restores verified cryptographic telecommand keys, restores T and E
                t_val = 1.0
                e_val = 1.0
                remaining_threats = [th for th in remaining_threats if th != "CADC_ANOMALY_SUSPECTED"]
                logger.info("[SelfHealingActuator] REVERT_TO_SAFE_TC_KEY_STORE applied: Restored T=1.0, E=1.0.")

        return StateVectorData(
            C=round(c_val, 4),
            R=round(r_val, 4),
            T=round(t_val, 4),
            Q=round(q_val, 4),
            M=round(m_val, 4),
            E=round(e_val, 4),
            A=round(a_val, 4),
            pace_state=s_t.pace_state,
            timestamp=s_t.timestamp,
            active_threats=remaining_threats,
        )

    # =================================================================
    # MODULE 4: Topology-Altering Actuators
    # These actuators modify the NetworkX graph structure to implement
    # the paper's Distributed Self-Healing requirement for compromised
    # satellite isolation and trusted route reconstruction.
    # =================================================================

    @staticmethod
    def isolate_compromised_node(graph: 'nx.DiGraph', node_id: str) -> List[tuple]:
        """
        Temporarily drops ALL active communication edges connected to the
        compromised satellite node in the CTIG graph. This prevents lateral
        threat propagation through the compromised node.
        
        Paper Reference: Stage 7 DSH — "compromised satellite isolation"
        
        Args:
            graph: NetworkX DiGraph (the CTIG constellation graph).
            node_id: ID of the satellite/node to isolate.
            
        Returns:
            List of removed edges (u, v, data) for potential later restoration.
        """
        import networkx as nx
        removed_edges = []
        if not graph.has_node(node_id):
            return removed_edges

        # Collect all edges touching this node
        edges_to_remove = []
        for u, v, data in graph.edges(data=True):
            if u == node_id or v == node_id:
                # Only remove communication edges, keep DEPENDS_ON for internal services
                relation = data.get("relation", "")
                if relation in ("COMMUNICATES_WITH", "communication_link", "TRUSTS", ""):
                    edges_to_remove.append((u, v, dict(data)))

        for u, v, data in edges_to_remove:
            graph.remove_edge(u, v)
            removed_edges.append((u, v, data))

        # Mark node as ISOLATED
        if graph.has_node(node_id):
            graph.nodes[node_id]["status"] = "ISOLATED"

        logger.info("[Stage 7 Actuator] ISOLATED node '%s': removed %d communication edges.",
                    node_id, len(removed_edges))
        return removed_edges

    @staticmethod
    def reconstruct_trusted_route(
        graph: 'nx.DiGraph', source: str, destination: str
    ) -> Optional[List[str]]:
        """
        Dynamically rebuilds the routing path between source and destination
        across remaining uncompromised nodes using Dijkstra shortest path
        weighted by inverse edge trust/quality.
        
        Paper Reference: Stage 7 DSH — "trusted route reconstruction"
        
        Args:
            graph: NetworkX DiGraph (the CTIG constellation graph).
            source: Source satellite node ID.
            destination: Destination satellite node ID.
            
        Returns:
            List of node IDs forming the reconstructed trusted route,
            or None if no path exists through healthy nodes.
        """
        import networkx as nx
        if not graph.has_node(source) or not graph.has_node(destination):
            return None

        # Build a subgraph containing only HEALTHY and non-ISOLATED nodes
        healthy_nodes = [
            n for n, d in graph.nodes(data=True)
            if d.get("status") not in ("COMPROMISED", "DEGRADED", "ISOLATED")
               and d.get("type") != "THREAT_INDICATOR"
        ]
        # Always include source and destination even if degraded
        for node in [source, destination]:
            if node not in healthy_nodes and graph.has_node(node):
                healthy_nodes.append(node)

        subgraph = graph.subgraph(healthy_nodes)

        try:
            # Use inverse weight as distance (lower trust = higher cost)
            def weight_fn(u, v, data):
                return 1.0 / max(0.01, data.get("weight", 1.0))

            path = nx.dijkstra_path(subgraph, source, destination, weight=weight_fn)
            logger.info("[Stage 7 Actuator] Reconstructed trusted route: %s", " -> ".join(path))
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            logger.warning("[Stage 7 Actuator] No trusted route found from '%s' to '%s'.", source, destination)
            return None


class DistributedSelfHealing(BaseStage):
    """
    Stage 7: Distributed Self-Healing & Physical Execution Module.
    Inherits from BaseStage to enforce plug-and-play adaptability.
    """

    def __init__(self, actuator: Optional[SelfHealingActuator] = None):
        super().__init__(name="Stage7_Healing")
        self.actuator = actuator or SelfHealingActuator()
        self.current_pace_state: PaceState = PaceState.PRIMARY

    def execute(self, input_data: Union[OptimizationOutput, Dict[str, Any]]) -> HealingOutput:
        """
        Executes Stage 7 self-healing and PACE state transition.
        
        Args:
            input_data: OptimizationOutput dataclass or dictionary containing actions and state vector.
            
        Returns:
            HealingOutput: Standardized execution artifact.
        """
        start_time = time.time()
        s_t = None

        if isinstance(input_data, OptimizationOutput):
            actions = input_data.optimal_actions
            target_pace = input_data.target_pace_state
        elif isinstance(input_data, dict):
            actions = input_data.get("optimal_actions", [])
            s_t = input_data.get("state_vector")
            target_pace_str = input_data.get("target_pace_state", "PRIMARY")
            target_pace = PaceState(target_pace_str) if target_pace_str in PaceState.__members__ else PaceState.PRIMARY
        else:
            actions = ["MAINTAIN_NOMINAL_OPERATIONS"]
            target_pace = PaceState.PRIMARY

        logger.info("[Stage 7 Healing] Triggering %d self-healing actuators and updating PACE state.", len(actions))

        confirmations = {}
        for action in actions:
            success = self.actuator.execute_action(action)
            confirmations[action] = success

        # Apply state recovery shifts if StateVectorData is provided
        if s_t and isinstance(s_t, StateVectorData):
            healed_st = self.actuator.heal_state_vector(s_t, actions)
            healed_st.pace_state = target_pace

        # Update physical PACE fallback state
        self.current_pace_state = target_pace
        logger.info("[Stage 7 Healing] Satellite PACE State transitioned to: %s", self.current_pace_state.value)

        elapsed_ms = round((time.time() - start_time) * 1000.0, 2)

        return HealingOutput(
            status="SUCCESS",
            current_pace_state=self.current_pace_state,
            executed_actions=actions,
            execution_timeline_ms=elapsed_ms,
            action_confirmations=confirmations,
        )

    def process(self, mitigation_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility bridge returning dictionary representation."""
        output = self.execute(mitigation_plan)
        return {
            "status": output.status,
            "current_pace_state": output.current_pace_state.value,
            "executed_count": len(output.executed_actions),
            "executed_actions": output.executed_actions,
        }


# Backward compatibility aliases
Stage7Healing = DistributedSelfHealing
