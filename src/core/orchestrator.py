"""
HCCRO Master Orchestrator Engine
================================
Sequentially connects and executes the 8 stages of cognitive cyber resilience optimization
using standardized interfaces (BaseStage) and strongly typed dataclass artifacts.

Includes a functional 5-tick simulation dashboard in `if __name__ == '__main__':`
evaluating Nominal, Jamming, Spoofing, Protocol Telecommand Injection / DoS,
and Self-Healed recovery ticks.
"""

import time
from typing import Dict, Any, Union
from src.core.interfaces import (
    TelemetryData, StateVectorData, CTIGOutput, AIMOutput,
    AEPOutput, MIAOutput, OptimizationOutput, HealingOutput, CKEOutput, PaceState
)
from src.core.state_vector import StateVector
from src.stages.stage1_csa import CyberSituationAwareness
from src.stages.stage2_ctig import CognitiveThreatIntelligenceGraph
from src.stages.stage3_aim import AttackIntentionModeling
from src.stages.stage4_aep import AttackEvolutionPrediction
from src.stages.stage5_mia import MissionImpactEstimation
from src.stages.stage6_optimization import Stage6Optimization
from src.stages.stage7_healing import Stage7Healing
from src.stages.stage8_cke import CyberKnowledgeEvolution
from src.utils.logger import get_logger

logger = get_logger("HCCRO.Orchestrator")


class HCCROOrchestrator:
    """
    Main orchestration engine executing the 8-stage HCCRO resilience loop.
    Enforces strict separation of concerns via standardized BaseStage execution contracts.
    """

    def __init__(self):
        logger.info("Initializing HCCRO 8-Stage Pipeline components...")
        self.stage1_csa = CyberSituationAwareness()
        self.stage2_ctig = CognitiveThreatIntelligenceGraph()
        self.stage3_aim = AttackIntentionModeling()
        self.stage4_aep = AttackEvolutionPrediction()
        self.stage5_mia = MissionImpactEstimation()
        self.stage6_opt = Stage6Optimization()
        self.stage7_heal = Stage7Healing()
        self.stage8_cke = CyberKnowledgeEvolution()

    def execute_pipeline(self, raw_telemetry: Union[TelemetryData, Dict[str, Any], None] = None, telemetry_data: Union[TelemetryData, Dict[str, Any], None] = None) -> Dict[str, Any]:
        """
        Executes one complete iteration of the 8-stage cognitive resilience loop using strongly-typed dataclasses.
        """
        data = raw_telemetry if raw_telemetry is not None else telemetry_data
        return self._run_pipeline_impl(data)

    # Alias for pipeline execution
    run_pipeline = execute_pipeline

    def _run_pipeline_impl(self, raw_telemetry: Union[TelemetryData, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Args:
            raw_telemetry: TelemetryData object or raw data dictionary.
            
        Returns:
            Dict containing pipeline results, optimal response plan, and updated state vector.
        """
        logger.info("--- Starting HCCRO Dataclass Pipeline Execution Cycle ---")
        
        # Stage 1: Cyber Situation Awareness (TelemetryData -> StateVectorData S_t)
        s_t: StateVectorData = self.stage1_csa.execute(raw_telemetry)
        
        # Stage 2: Cognitive Threat Intelligence Graph Construction (StateVectorData -> CTIGOutput)
        ctig_out: CTIGOutput = self.stage2_ctig.execute({"state_vector": s_t, "telemetry": raw_telemetry})
        
        # Stage 3: Attack Intention Modeling (StateVectorData & CTIGOutput -> AIMOutput)
        aim_out: AIMOutput = self.stage3_aim.execute({
            "state_vector": s_t,
            "ctig_output": ctig_out
        })
        
        # Stage 4: Attack Evolution Prediction (AIMOutput & CTIGOutput -> AEPOutput)
        aep_out: AEPOutput = self.stage4_aep.execute({
            "aim_output": aim_out,
            "ctig_output": ctig_out
        })
        
        # Stage 5: Mission Impact Estimation (StateVectorData & AEPOutput -> MIAOutput)
        mia_out: MIAOutput = self.stage5_mia.execute({
            "state_vector": s_t,
            "aep_output": aep_out
        })
        
        # Stage 6: Multi-Objective Optimization (StateVectorData & MIAOutput -> OptimizationOutput)
        opt_out: OptimizationOutput = self.stage6_opt.execute({
            "state_vector": s_t,
            "mia_output": mia_out
        })
        
        # Stage 7: Distributed Self-Healing Execution (OptimizationOutput -> HealingOutput)
        heal_out: HealingOutput = self.stage7_heal.execute(opt_out)
        
        # Update s_t pace_state to reflect self-healing outcome
        s_t.pace_state = heal_out.current_pace_state

        # Stage 8: Cyber Knowledge Evolution Logging (StateVectorData, OptimizationOutput, HealingOutput -> CKEOutput)
        cke_out: CKEOutput = self.stage8_cke.execute({
            "state_vector": s_t,
            "optimization_output": opt_out,
            "healing_output": heal_out
        })
        
        logger.info("--- Cycle Complete: Mitigation status=%s (PACE State: %s) ---", heal_out.status, heal_out.current_pace_state.value)
        
        return {
            "state_vector": s_t,
            "ctig_output": ctig_out,
            "aim_output": aim_out,
            "aep_output": aep_out,
            "impact_metrics": mia_out,
            "mitigation_plan": opt_out,
            "execution_status": heal_out,
            "knowledge_record": cke_out,
        }

    def run_cycle(self, raw_telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility interface returning dictionary representation."""
        res = self.execute_pipeline(raw_telemetry)
        return {
            "state_vector": res["state_vector"].to_dict(),
            "pace_state": res["state_vector"].pace_state.value,
            "impact_metrics": {
                "mci_score": res["impact_metrics"].mci_score,
                "mdi_score": res["impact_metrics"].mdi_score,
                "threat_severity": res["impact_metrics"].threat_severity,
                "impacted_subsystems": res["impact_metrics"].affected_subsystems,
            },
            "mitigation_plan": {
                "optimal_actions": res["mitigation_plan"].optimal_actions,
                "target_pace_state": res["mitigation_plan"].target_pace_state.value,
                "solver_objective_value": res["mitigation_plan"].utility_score,
                "drei_score": res["mitigation_plan"].drei_score,
                "weights_used": res["mitigation_plan"].weights_applied,
            },
            "execution_status": {
                "status": res["execution_status"].status,
                "current_pace_state": res["execution_status"].current_pace_state.value,
                "executed_count": len(res["execution_status"].executed_actions),
                "executed_actions": res["execution_status"].executed_actions,
            },
            "knowledge_record": {
                "record_id": res["knowledge_record"].record_id,
                "threats_logged": res["state_vector"].active_threats,
                "success_flag": res["knowledge_record"].persisted_successfully,
            },
        }


# =============================================================================
# Functional End-to-End Simulation Dashboard (5 Consecutive Ticks)
# =============================================================================
if __name__ == "__main__":
    print("========================================================================================")
    print("      HIERARCHICAL COGNITIVE CYBER RESILIENCE OPTIMIZATION (HCCRO) DASHBOARD SIMULATION     ")
    print("========================================================================================")

    orchestrator = HCCROOrchestrator()

    # Define 5 Consecutive Simulation Ticks
    simulation_ticks = [
        {
            "tick": 1,
            "name": "Tick 1: Nominal Operations",
            "telemetry": {
                "packet_delivery_ratio": 1.0,
                "communication_latency": 12.0,
                "spectrum_usage": 0.04,
                "navigation_consistency": 0.5,
                "authentication_events": 0,
                "processor_utilization": 15.0,
                "memory_consumption": 20.0,
                "energy_availability": 0.98,
                "software_integrity_measurements": 1.0,
            },
        },
        {
            "tick": 2,
            "name": "Tick 2: RF Jamming Anomaly Event",
            "telemetry": {
                "packet_delivery_ratio": 0.20,
                "communication_latency": 450.0,
                "spectrum_usage": 0.85,
                "navigation_consistency": 1.0,
                "authentication_events": 0,
                "processor_utilization": 22.0,
                "memory_consumption": 25.0,
                "energy_availability": 0.95,
                "software_integrity_measurements": 1.0,
            },
        },
        {
            "tick": 3,
            "name": "Tick 3: GNSS Spoofing Anomaly Event",
            "telemetry": {
                "packet_delivery_ratio": 0.95,
                "communication_latency": 18.0,
                "spectrum_usage": 0.05,
                "navigation_consistency": 45.0,
                "authentication_events": 5,
                "processor_utilization": 25.0,
                "memory_consumption": 28.0,
                "energy_availability": 0.90,
                "software_integrity_measurements": 0.98,
            },
        },
        {
            "tick": 4,
            "name": "Tick 4: Resource Exhaustion DoS & Battery Depletion Event",
            "telemetry": {
                "packet_delivery_ratio": 0.92,
                "communication_latency": 25.0,
                "spectrum_usage": 0.06,
                "navigation_consistency": 1.2,
                "authentication_events": 0,
                "processor_utilization": 99.0,
                "memory_consumption": 95.0,
                "energy_availability": 0.35,  # Critical Battery < 40%!
                "software_integrity_measurements": 0.95,
            },
        },
        {
            "tick": 5,
            "name": "Tick 5: Self-Healed & Recharged Operational State",
            "telemetry": {
                "packet_delivery_ratio": 0.99,
                "communication_latency": 15.0,
                "spectrum_usage": 0.05,
                "navigation_consistency": 0.8,
                "authentication_events": 0,
                "processor_utilization": 18.0,
                "memory_consumption": 22.0,
                "energy_availability": 0.92,
                "software_integrity_measurements": 1.0,
            },
        },
    ]

    for step in simulation_ticks:
        t_num = step["tick"]
        t_name = step["name"]
        raw_tele = step["telemetry"]

        res = orchestrator.execute_pipeline(raw_tele)

        st: StateVectorData = res["state_vector"]
        aim: AIMOutput = res["aim_output"]
        aep: AEPOutput = res["aep_output"]
        mia: MIAOutput = res["impact_metrics"]
        opt: OptimizationOutput = res["mitigation_plan"]
        heal: HealingOutput = res["execution_status"]

        print("\n" + "-" * 88)
        print(f"  >>> {t_name.upper()} <<<")
        print("-" * 88)
        print(f"  [1] Operational State Vector S_t: C={st.C:.2f} | R={st.R:.2f} | T={st.T:.2f} | Q={st.Q:.2f} | M={st.M:.2f} | E={st.E:.2f} | A={st.A:.2f}")
        print(f"      Active Threat Flags           : {st.active_threats if st.active_threats else '[] (NOMINAL)'}")
        print(f"  [2] Onboard PACE Edge Sub-layer  : ACTIVE PACE STATE = [{heal.current_pace_state.value}] (Target: {opt.target_pace_state.value})")
        print(f"  [3] Threat Intention & Propagation: Primary Intent = {aim.primary_intention} (Conf: {aim.confidence_level*100:.0f}%)")
        print(f"      Lateral Cascade Probs         : {aep.lateral_propagation_probabilities}")
        print(f"      Mission Degradation Index     : MDI={mia.mdi_score:.3f} | Severity={mia.threat_severity} | MCI={mia.mci_score:.3f}")
        print(f"  [4] Multi-Objective Optimization : Solver Utility = {opt.utility_score:.4f} | DREI Index = {opt.drei_score:.4f}")
        print(f"      Adaptive Weights Applied      : {opt.weights_applied}")
        print(f"  [5] Self-Healing Execution Plan   : Actions = {heal.executed_actions}")
        print(f"      Actuator Status               : {heal.status} (Execution Time: {heal.execution_timeline_ms:.2f} ms)")

    print("\n========================================================================================")
    print("      5-TICK HCCRO FRAMEWORK SIMULATION COMPLETED SUCCESSFULLY WITH CLEAN RECOVERY     ")
    print("========================================================================================")
