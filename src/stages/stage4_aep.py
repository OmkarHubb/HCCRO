"""
Stage 4: Attack Evolution Prediction (AEP)
=========================================
Forecasts lateral threat movement and cascading failure propagation trajectories across
neighboring satellite nodes using an Independent Cascade Model (ICM) and Markov transitions.
"""

from typing import Dict, Any, List
import networkx as nx

from src.core.interfaces import BaseStage, AIMOutput, CTIGOutput, AEPOutput
from src.utils.logger import get_logger

logger = get_logger("Stage4.AEP")


class AttackEvolutionPrediction(BaseStage):
    """
    Stage 4: Attack Evolution Prediction (AEP) Module.
    Inherits from BaseStage to enforce plug-and-play adaptability.
    """

    def __init__(self):
        super().__init__(name="Stage4_AEP")

    def _independent_cascade_model(
        self, aim: AIMOutput, ctig: CTIGOutput
    ) -> Dict[str, float]:
        """
        Probabilistic Independent Cascade Model (ICM) estimating threat propagation
        likelihood across neighboring satellite nodes over inter-satellite links (ISLs).
        """
        graph = ctig.graph if ctig and ctig.graph else nx.DiGraph()
        lateral_probs: Dict[str, float] = {}

        # Default constellation satellites
        neighbor_sats = ["SAT_LEO_02", "SAT_LEO_03"]

        primary_intent_score = max(aim.intention_scores.values()) if aim.intention_scores else 0.0

        for sat in neighbor_sats:
            if ctig.compromised_nodes:
                # Base cascade probability weighted by active intent score and edge trust
                trust_weight = 0.85
                if graph.has_edge("SAT_LEO_01", sat):
                    trust_weight = graph["SAT_LEO_01"][sat].get("weight", 0.85)
                prob = round(primary_intent_score * trust_weight * 0.90, 3)
            else:
                prob = 0.05
            lateral_probs[sat] = min(1.0, max(0.0, prob))

        return lateral_probs

    def _evolution_predictor_slot(self, aim: AIMOutput, ctig: CTIGOutput) -> AEPOutput:
        """
        Pluggable lateral propagation prediction slot integrating ICM.
        """
        predicted_paths = []
        max_risk = 0.0

        lateral_probs = self._independent_cascade_model(aim, ctig)

        for node in ctig.compromised_nodes:
            risk = round(aim.intention_scores.get(aim.primary_intention, 0.5) * 0.9, 3)
            max_risk = max(max_risk, risk)
            for neighbor, prob in lateral_probs.items():
                predicted_paths.append({
                    "source": node,
                    "target": neighbor,
                    "estimated_propagation_time_sec": 45.0,
                    "escalation_probability": prob,
                })

        if not predicted_paths:
            predicted_paths.append({
                "source": "NOMINAL",
                "target": "NOMINAL",
                "estimated_propagation_time_sec": 0.0,
                "escalation_probability": 0.0,
            })

        return AEPOutput(
            predicted_paths=predicted_paths,
            lateral_propagation_probabilities=lateral_probs,
            escalation_risk=max_risk,
        )

    def execute(self, input_data: Dict[str, Any]) -> AEPOutput:
        """
        Executes Stage 4 evolution prediction using standard interfaces.
        
        Args:
            input_data: Dict containing 'aim_output' (AIMOutput) and 'ctig_output' (CTIGOutput).
            
        Returns:
            AEPOutput: Standardized attack evolution artifact.
        """
        aim = input_data.get("aim_output") or AIMOutput({"NOMINAL": 0.0}, "NOMINAL", 1.0)
        ctig = input_data.get("ctig_output") or CTIGOutput(nx.DiGraph(), [], [], 0)

        logger.info("[Stage 4 AEP] Predicting Independent Cascade Model lateral threat propagation.")
        return self._evolution_predictor_slot(aim, ctig)

    def process(self, intentions: Any, graph_data: Any) -> List[Dict[str, Any]]:
        """Backward compatibility bridge returning list of path dicts."""
        compromised = graph_data.get("compromised_nodes", []) if isinstance(graph_data, dict) else []
        ctig = CTIGOutput(nx.DiGraph(), compromised, [], 0)
        aim = AIMOutput(intentions.get("intentions", {}) if isinstance(intentions, dict) else {}, "NOMINAL", 1.0)
        output = self._evolution_predictor_slot(aim, ctig)
        return output.predicted_paths


# Backward compatibility alias
Stage4AEP = AttackEvolutionPrediction
