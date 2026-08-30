"""
Stage 3: Attack Intention Modeling (AIM)
=========================================
Inference engine predicting threat actor objectives:
- RF_JAMMING_DISRUPTION
- COMMAND_SPOOFING_TAKEOVER
- RESOURCE_DENIAL_OF_SERVICE

Exposes an extensible PredictiveModelRegistry to register machine learning models
with a structural graph topology fallback heuristic.
"""

from typing import Dict, Any, Union, Callable, Optional
import networkx as nx

from src.core.interfaces import BaseStage, StateVectorData, CTIGOutput, AIMOutput
from src.utils.logger import get_logger

logger = get_logger("Stage3.AIM")


class PredictiveModelRegistry:
    """
    Extensible model registry for registering machine learning classifiers
    (e.g., PyTorch, Scikit-learn, Bayesian Networks) for attack intention inference.
    """

    def __init__(self):
        self._registry: Dict[str, Callable[[CTIGOutput, StateVectorData], float]] = {}

    def register_model(
        self, intent_category: str, model_func: Callable[[CTIGOutput, StateVectorData], float]
    ) -> None:
        """Registers a custom ML inference function for a specific attack category."""
        self._registry[intent_category] = model_func
        logger.info("[AIM Registry] Registered custom ML model for intent '%s'.", intent_category)

    def has_model(self, intent_category: str) -> bool:
        """Checks if a custom model function is registered for the category."""
        return intent_category in self._registry

    def predict(self, intent_category: str, ctig: CTIGOutput, s_t: StateVectorData) -> Optional[float]:
        """Executes registered model function if available."""
        if intent_category in self._registry:
            try:
                return float(self._registry[intent_category](ctig, s_t))
            except Exception as err:
                logger.warning("[AIM Registry] Model inference error for '%s': %s", intent_category, err)
        return None


class AttackIntentionModeling(BaseStage):
    """
    Stage 3: Attack Intention Modeling (AIM) Engine.
    Inherits from BaseStage to enforce plug-and-play adaptability.
    """

    def __init__(self, registry: Optional[PredictiveModelRegistry] = None):
        super().__init__(name="Stage3_AIM")
        self.registry = registry or PredictiveModelRegistry()

    def _structural_fallback_heuristic(self, ctig: CTIGOutput, s_t: StateVectorData) -> Dict[str, float]:
        """
        Structural fallback heuristic analyzing graph motifs and shortest path connectivity
        between injected THREAT_INDICATOR vertices and critical services/links.
        """
        graph = ctig.graph if ctig and ctig.graph else nx.DiGraph()

        jamming_score = 0.10
        spoofing_score = 0.05
        dos_score = 0.10

        # Check graph nodes and edges for threat indicators
        threat_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("type") == "THREAT_INDICATOR"
        ]

        for threat_node in threat_nodes:
            node_data = graph.nodes[threat_node]
            subtype = node_data.get("subtype", "")

            if subtype == "RF_JAMMING" or "JAMMING" in threat_node:
                jamming_score = max(jamming_score, 0.92)
            elif subtype == "GNSS_SPOOFING" or "SPOOFING" in threat_node:
                spoofing_score = max(spoofing_score, 0.88)
            elif subtype == "RESOURCE_EXHAUSTION" or "RESOURCE" in threat_node:
                dos_score = max(dos_score, 0.95)

        # Fallback to S_t vector indicators if graph has no threat nodes
        if "RF_JAMMING_SUSPECTED" in s_t.active_threats or s_t.C < 0.5:
            jamming_score = max(jamming_score, 0.90)
        if "GPS_SPOOFING_SUSPECTED" in s_t.active_threats or s_t.T < 0.5:
            spoofing_score = max(spoofing_score, 0.85)
        if "RESOURCE_DOS_SUSPECTED" in s_t.active_threats or s_t.E < 0.2:
            dos_score = max(dos_score, 0.95)

        return {
            "RF_JAMMING_DISRUPTION": round(jamming_score, 4),
            "COMMAND_SPOOFING_TAKEOVER": round(spoofing_score, 4),
            "RESOURCE_DENIAL_OF_SERVICE": round(dos_score, 4),
        }

    def _inference_engine_slot(self, s_t: StateVectorData, ctig: CTIGOutput) -> AIMOutput:
        """
        Inference engine executing registered ML models with structural fallback.
        """
        categories = ["RF_JAMMING_DISRUPTION", "COMMAND_SPOOFING_TAKEOVER", "RESOURCE_DENIAL_OF_SERVICE"]
        intentions = {}

        fallback_scores = self._structural_fallback_heuristic(ctig, s_t)

        for cat in categories:
            if self.registry.has_model(cat):
                pred = self.registry.predict(cat, ctig, s_t)
                intentions[cat] = round(pred if pred is not None else fallback_scores[cat], 4)
            else:
                intentions[cat] = fallback_scores[cat]

        primary_intent = max(intentions, key=intentions.get)
        max_score = intentions[primary_intent]
        confidence = 0.95 if max_score > 0.5 else 0.40

        return AIMOutput(
            intention_scores=intentions,
            primary_intention=primary_intent,
            confidence_level=confidence,
        )

    def execute(self, input_data: Dict[str, Any]) -> AIMOutput:
        """
        Executes Stage 3 intention inference using standard interfaces.
        
        Args:
            input_data: Dict containing 'state_vector' (StateVectorData) and 'ctig_output' (CTIGOutput).
            
        Returns:
            AIMOutput: Standardized attack intention artifact.
        """
        s_t = input_data.get("state_vector") or StateVectorData()
        ctig = input_data.get("ctig_output") or CTIGOutput(nx.DiGraph(), [], [], 0)

        logger.info("[Stage 3 AIM] Evaluating Attack Intention Modeling across threat graph.")
        return self._inference_engine_slot(s_t, ctig)

    def process(self, state: Any, graph_data: Any) -> Dict[str, Any]:
        """Backward compatibility bridge returning dictionary representation."""
        if isinstance(state, StateVectorData):
            s_t = state
        else:
            s_t = StateVectorData(
                C=getattr(state, "s_t", {}).get("C", 1.0),
                T=getattr(state, "s_t", {}).get("T", 1.0),
                E=getattr(state, "s_t", {}).get("E", 1.0),
                active_threats=getattr(state, "active_threats", []),
            )
        ctig = CTIGOutput(nx.DiGraph(), graph_data.get("compromised_nodes", []) if isinstance(graph_data, dict) else [], [], 0)
        output = self._inference_engine_slot(s_t, ctig)
        return {"intentions": output.intention_scores, "primary_intent": output.primary_intention}


# Backward compatibility alias
Stage3AIM = AttackIntentionModeling
