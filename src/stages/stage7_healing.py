"""
Stage 7: Distributed Self-Healing (DSH)
========================================
Executes selected countermeasure actions using a SelfHealingActuator callback registry
and updates physical satellite PACE fallback sub-layer states.
"""

import time
from typing import Dict, Any, Callable, List, Optional, Union
from src.core.interfaces import BaseStage, OptimizationOutput, HealingOutput, PaceState
from src.utils.logger import get_logger

logger = get_logger("Stage7.Healing")


class SelfHealingActuator:
    """
    Actuator registry mapping strategic action strings directly to physical execution callbacks.
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
        logger.info("[Stage 7 Actuator] Simulated physical execution of command: '%s'", action_key)
        return True

    def _register_default_actuators(self) -> None:
        """Registers default actuator callbacks for core resilience actions."""
        actions = [
            "ACTIVATE_FREQUENCY_HOPPING",
            "SWITCH_TO_INERTIAL_NAV_FALLBACK",
            "ENFORCE_PROCESS_QUOTA_ISOLATION",
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


class DistributedSelfHealing(BaseStage):
    """
    Stage 7: Distributed Self-Healing & PACE Execution Module.
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
            input_data: OptimizationOutput dataclass or dictionary.
            
        Returns:
            HealingOutput: Standardized execution artifact.
        """
        start_time = time.time()

        if isinstance(input_data, OptimizationOutput):
            actions = input_data.optimal_actions
            target_pace = input_data.target_pace_state
        elif isinstance(input_data, dict):
            actions = input_data.get("optimal_actions", [])
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


# Backward compatibility alias
Stage7Healing = DistributedSelfHealing
