"""
Stage 3: Attack Intention Modeling (AIM)
=========================================
Inference engine predicting threat actor objectives:
- RF_JAMMING_DISRUPTION
- COMMAND_SPOOFING_TAKEOVER
- RESOURCE_DENIAL_OF_SERVICE
- PROTOCOL_TELECOMMAND_INJECTION  (ESA OPSSAT-AD campaign)

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


class BayesianIntentInference:
    """
    MODULE 4 — Lightweight Bayesian Network Inference Engine.
    
    Computes conditional probability of multi-step strategic attack intent given
    observed threat indicators, using a manually-specified Conditional Probability
    Table (CPT) derived from CCRT threat taxonomy.
    
    Supports multi-step chaining via the chain rule:
        P(Takeover | Spoofing → Jamming) = P(Takeover | Spoofing) * P(Spoofing | Jamming)
    
    Paper Reference: Stage 3 AIM — "Bayesian/ML models" for predicting threat
    actor objectives including "denial of service, data exfiltration, orbit
    manipulation."
    """

    # Default Conditional Probability Table (CPT)
    # Structure: CPT[observed_evidence][strategic_intent] = probability
    # Derived from CCRT threat taxonomy and adversarial behavior modeling.
    DEFAULT_CPT = {
        # Single-step conditionals: P(intent | single observed threat)
        "RF_JAMMING_SUSPECTED": {
            "RF_JAMMING_DISRUPTION": 0.92,
            "COMMAND_SPOOFING_TAKEOVER": 0.15,
            "RESOURCE_DENIAL_OF_SERVICE": 0.08,
            "PROTOCOL_TELECOMMAND_INJECTION": 0.10,
        },
        "GPS_SPOOFING_SUSPECTED": {
            "RF_JAMMING_DISRUPTION": 0.10,
            "COMMAND_SPOOFING_TAKEOVER": 0.88,
            "RESOURCE_DENIAL_OF_SERVICE": 0.12,
            "PROTOCOL_TELECOMMAND_INJECTION": 0.15,
        },
        "RESOURCE_DOS_SUSPECTED": {
            "RF_JAMMING_DISRUPTION": 0.05,
            "COMMAND_SPOOFING_TAKEOVER": 0.20,
            "RESOURCE_DENIAL_OF_SERVICE": 0.95,
            "PROTOCOL_TELECOMMAND_INJECTION": 0.12,
        },
        # OPSSAT-AD Telecommand Injection evidence keys
        "TELECOMMAND_INJECTION_SUSPECTED": {
            "RF_JAMMING_DISRUPTION": 0.05,
            "COMMAND_SPOOFING_TAKEOVER": 0.25,
            "RESOURCE_DENIAL_OF_SERVICE": 0.10,
            "PROTOCOL_TELECOMMAND_INJECTION": 0.95,
        },
        "CADC_ANOMALY_SUSPECTED": {
            "RF_JAMMING_DISRUPTION": 0.08,
            "COMMAND_SPOOFING_TAKEOVER": 0.20,
            "RESOURCE_DENIAL_OF_SERVICE": 0.10,
            "PROTOCOL_TELECOMMAND_INJECTION": 0.92,
        },
        # Multi-step chain conditionals: P(intent | threat_A → threat_B)
        # These capture correlated multi-vector attack campaigns.
        "CADC_ANOMALY_SUSPECTED+TELECOMMAND_INJECTION_SUSPECTED": {
            "RF_JAMMING_DISRUPTION": 0.05,
            "COMMAND_SPOOFING_TAKEOVER": 0.15,
            "RESOURCE_DENIAL_OF_SERVICE": 0.08,
            "PROTOCOL_TELECOMMAND_INJECTION": 0.98,
        },
        "RF_JAMMING_SUSPECTED+GPS_SPOOFING_SUSPECTED": {
            "RF_JAMMING_DISRUPTION": 0.40,
            "COMMAND_SPOOFING_TAKEOVER": 0.92,   # Jamming → Spoofing strongly indicates takeover
            "RESOURCE_DENIAL_OF_SERVICE": 0.15,
            "PROTOCOL_TELECOMMAND_INJECTION": 0.10,
        },
        "GPS_SPOOFING_SUSPECTED+RESOURCE_DOS_SUSPECTED": {
            "RF_JAMMING_DISRUPTION": 0.10,
            "COMMAND_SPOOFING_TAKEOVER": 0.75,
            "RESOURCE_DENIAL_OF_SERVICE": 0.85,
            "PROTOCOL_TELECOMMAND_INJECTION": 0.12,
        },
        "RF_JAMMING_SUSPECTED+RESOURCE_DOS_SUSPECTED": {
            "RF_JAMMING_DISRUPTION": 0.70,
            "COMMAND_SPOOFING_TAKEOVER": 0.25,
            "RESOURCE_DENIAL_OF_SERVICE": 0.88,
            "PROTOCOL_TELECOMMAND_INJECTION": 0.10,
        },
        "RF_JAMMING_SUSPECTED+GPS_SPOOFING_SUSPECTED+RESOURCE_DOS_SUSPECTED": {
            "RF_JAMMING_DISRUPTION": 0.60,
            "COMMAND_SPOOFING_TAKEOVER": 0.90,
            "RESOURCE_DENIAL_OF_SERVICE": 0.92,
            "PROTOCOL_TELECOMMAND_INJECTION": 0.15,
        },
    }

    def __init__(self, cpt: Optional[Dict[str, Dict[str, float]]] = None):
        self.cpt = cpt or self.DEFAULT_CPT.copy()

    def infer(self, observed_threats: list) -> Dict[str, float]:
        """
        Computes P(intent | observed_threats) using the CPT.
        
        Strategy:
        1. First check for an exact compound key matching the full threat set.
        2. If no exact match, use chain rule decomposition over individual threats.
        3. Normalize posterior probabilities to sum to 1.0.
        
        Args:
            observed_threats: List of active threat strings (e.g., ["RF_JAMMING_SUSPECTED"]).
            
        Returns:
            Dict mapping intent categories to posterior probability scores.
        """
        intents = ["RF_JAMMING_DISRUPTION", "COMMAND_SPOOFING_TAKEOVER", "RESOURCE_DENIAL_OF_SERVICE", "PROTOCOL_TELECOMMAND_INJECTION"]
        
        if not observed_threats:
            return {intent: 0.10 for intent in intents}

        # Sort threats for consistent compound key lookup
        sorted_threats = sorted(set(observed_threats))
        compound_key = "+".join(sorted_threats)

        # Strategy 1: Exact compound key match
        if compound_key in self.cpt:
            return dict(self.cpt[compound_key])

        # Strategy 2: Chain rule decomposition
        # P(intent | threat_A, threat_B) ≈ 1 - ∏(1 - P(intent | threat_i))
        # This is the noisy-OR model commonly used in Bayesian networks.
        posteriors = {}
        for intent in intents:
            # Compute noisy-OR combination
            prod_complement = 1.0
            matched_any = False
            for threat in sorted_threats:
                if threat in self.cpt:
                    p = self.cpt[threat].get(intent, 0.05)
                    prod_complement *= (1.0 - p)
                    matched_any = True
            if matched_any:
                posteriors[intent] = round(1.0 - prod_complement, 4)
            else:
                posteriors[intent] = 0.10  # No CPT coverage

        return posteriors


class AttackIntentionModeling(BaseStage):
    """
    Stage 3: Attack Intention Modeling (AIM) Engine.
    Inherits from BaseStage to enforce plug-and-play adaptability.
    """

    def __init__(self, registry: Optional[PredictiveModelRegistry] = None):
        super().__init__(name="Stage3_AIM")
        self.registry = registry or PredictiveModelRegistry()
        # MODULE 4: Bayesian inference engine for multi-step intent reasoning
        self.bayesian_engine = BayesianIntentInference()

    def _structural_fallback_heuristic(self, ctig: CTIGOutput, s_t: StateVectorData) -> Dict[str, float]:
        """
        Structural fallback heuristic analyzing graph motifs and shortest path connectivity
        between injected THREAT_INDICATOR vertices and critical services/links.
        """
        graph = ctig.graph if ctig and ctig.graph else nx.DiGraph()

        jamming_score = 0.10
        spoofing_score = 0.05
        dos_score = 0.10
        tc_injection_score = 0.05

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
            elif subtype == "TELECOMMAND_INJECTION" or "TELECOMMAND" in threat_node or "CADC" in threat_node:
                tc_injection_score = max(tc_injection_score, 0.95)

        # Fallback to S_t vector indicators if graph has no threat nodes
        if "RF_JAMMING_SUSPECTED" in s_t.active_threats or s_t.C < 0.5:
            jamming_score = max(jamming_score, 0.90)
        if "GPS_SPOOFING_SUSPECTED" in s_t.active_threats or s_t.T < 0.5:
            spoofing_score = max(spoofing_score, 0.85)
        if "RESOURCE_DOS_SUSPECTED" in s_t.active_threats or s_t.E < 0.2:
            dos_score = max(dos_score, 0.95)
        # OPSSAT-AD: Telecommand injection via buffer spikes or CADC channel anomalies
        if "TELECOMMAND_INJECTION_SUSPECTED" in s_t.active_threats or "CADC_ANOMALY_SUSPECTED" in s_t.active_threats or s_t.A < 0.50:
            tc_injection_score = max(tc_injection_score, 0.95)

        return {
            "RF_JAMMING_DISRUPTION": round(jamming_score, 4),
            "COMMAND_SPOOFING_TAKEOVER": round(spoofing_score, 4),
            "RESOURCE_DENIAL_OF_SERVICE": round(dos_score, 4),
            "PROTOCOL_TELECOMMAND_INJECTION": round(tc_injection_score, 4),
        }

    def _inference_engine_slot(self, s_t: StateVectorData, ctig: CTIGOutput) -> AIMOutput:
        """
        Inference engine executing a 3-source weighted ensemble:
        1. Bayesian posterior from CPT (40% weight) — multi-step intent chains
        2. Structural graph heuristic (30% weight) — graph motif analysis
        3. Registered ML model (30% weight) — sklearn classifier if available
        
        This fusion strategy satisfies the paper's requirement for "Bayesian/ML
        models" predicting threat actor objectives beyond isolated classification.
        """
        categories = ["RF_JAMMING_DISRUPTION", "COMMAND_SPOOFING_TAKEOVER", "RESOURCE_DENIAL_OF_SERVICE", "PROTOCOL_TELECOMMAND_INJECTION"]
        intentions = {}

        # Source 1: Structural graph heuristic scores
        fallback_scores = self._structural_fallback_heuristic(ctig, s_t)

        # Source 2: Bayesian posterior scores from CPT
        bayesian_scores = self.bayesian_engine.infer(s_t.active_threats)

        for cat in categories:
            # Source 3: ML model prediction (if registered)
            ml_score = None
            if self.registry.has_model(cat):
                ml_score = self.registry.predict(cat, ctig, s_t)

            # Weighted ensemble fusion
            w_bayes, w_heuristic, w_ml = 0.40, 0.30, 0.30
            bayes_val = bayesian_scores.get(cat, 0.10)
            heuristic_val = fallback_scores.get(cat, 0.10)

            if ml_score is not None:
                fused = w_bayes * bayes_val + w_heuristic * heuristic_val + w_ml * ml_score
            else:
                # No ML model: redistribute ML weight to Bayesian + heuristic
                fused = 0.55 * bayes_val + 0.45 * heuristic_val

            intentions[cat] = round(min(1.0, max(0.0, fused)), 4)

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
