"""
src/stages/stage3_aim.py

Stage 3 -- Attack Intention Modeling (AIM)

- `PredictiveModelRegistry`: decoupling layer for swapping in any trained
  scikit-learn-compatible classifier (or a thin wrapper around a PyTorch
  model exposing the same .predict()/.predict_proba() interface).
- `AttackIntentionModeler`: consumes the Stage 2 CTIG + the current 7-D
  Operational State Vector St = {C,R,T,Q,M,E,A} -- NOT raw telemetry -- and
  returns exactly one of the four standardized intent strings below. Raw
  telemetry (C/N0, GPS drift, CPU%, etc.) is Stage 1/Stage 2's concern;
  by the time it reaches Stage 3 it has already been normalized into St,
  which is also what Stage 6's optimizer and the SQLite incident log
  (Stage 8) key off of -- so St is the only input contract Stage 3 needs.

Standardized output labels (must match Stage 6 / SQLite logging exactly):
    'NOMINAL'
    'RF_JAMMING_DISRUPTION'
    'COMMAND_SPOOFING_TAKEOVER'
    'RESOURCE_DENIAL_OF_SERVICE'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import numpy as np

from src.stages.stage2_ctig import CognitiveThreatGraph
from src.utils.state_vector import StateVector, coerce_state_vector

INTENTS = [
    "NOMINAL",
    "RF_JAMMING_DISRUPTION",
    "COMMAND_SPOOFING_TAKEOVER",
    "RESOURCE_DENIAL_OF_SERVICE",
]

# Map Stage 2's THREAT_INDICATOR subtypes onto Stage 3's inferred intents,
# used both as the heuristic fallback and to label training data.
SUBTYPE_TO_INTENT = {
    "JAMMING": "RF_JAMMING_DISRUPTION",
    "SPOOFING": "COMMAND_SPOOFING_TAKEOVER",
    "DoS": "RESOURCE_DENIAL_OF_SERVICE",
}

FEATURE_NAMES = ["C", "R", "T", "Q", "M", "E", "A"]


def extract_features(state: Union[StateVector, dict]) -> np.ndarray:
    """The classifier's ONLY input: the 7-D Operational State Vector, in the
    fixed order [C, R, T, Q, M, E, A]. Accepts a StateVector or a plain
    dict (Input Contract)."""
    sv = coerce_state_vector(state)
    return np.array([sv.C, sv.R, sv.T, sv.Q, sv.M, sv.E, sv.A], dtype=float)


@dataclass
class RegisteredModel:
    name: str
    model: object                      # exposes .predict(X) -> labels, .predict_proba optional
    label_encoder: Optional[object]    # exposes .inverse_transform if labels are encoded ints
    feature_fn: Callable[[Union[StateVector, dict]], np.ndarray] = extract_features


class PredictiveModelRegistry:
    """Holds zero or more classifiers and tracks which one is "active"."""

    def __init__(self) -> None:
        self._models: Dict[str, RegisteredModel] = {}
        self._active: Optional[str] = None

    def register(self, name: str, model: object, label_encoder: object = None,
                 feature_fn: Callable = extract_features, make_active: bool = True) -> None:
        self._models[name] = RegisteredModel(name, model, label_encoder, feature_fn)
        if make_active or self._active is None:
            self._active = name

    def set_active(self, name: str) -> None:
        if name not in self._models:
            raise KeyError(f"No model registered under '{name}'")
        self._active = name

    def has_active(self) -> bool:
        return self._active is not None

    def predict(self, state: Union[StateVector, dict]) -> Optional[str]:
        if self._active is None:
            return None
        entry = self._models[self._active]
        x = entry.feature_fn(state).reshape(1, -1)
        pred = entry.model.predict(x)[0]
        if entry.label_encoder is not None:
            pred = entry.label_encoder.inverse_transform([pred])[0]
        return str(pred)

    def predict_proba(self, state: Union[StateVector, dict]) -> Optional[Dict[str, float]]:
        if self._active is None or not hasattr(self._models[self._active].model, "predict_proba"):
            return None
        entry = self._models[self._active]
        x = entry.feature_fn(state).reshape(1, -1)
        proba = entry.model.predict_proba(x)[0]
        classes = entry.model.classes_
        if entry.label_encoder is not None:
            classes = entry.label_encoder.inverse_transform(classes)
        return {str(c): float(p) for c, p in zip(classes, proba)}


class AttackIntentionModeler:
    """Stage 3: infers adversary intent from the CTIG + the 7-D State Vector."""

    def __init__(self, registry: Optional[PredictiveModelRegistry] = None) -> None:
        self.registry = registry or PredictiveModelRegistry()

    def infer(self, ctig: CognitiveThreatGraph, satellite_id: str,
              state: Union[StateVector, dict]) -> str:
        state = coerce_state_vector(state)

        # 1) Prefer the registered ML classifier, if one is active.
        if self.registry.has_active():
            try:
                intent = self.registry.predict(state)
                if intent in INTENTS:
                    return intent
            except Exception:
                pass  # fall through to heuristic below

        # 2) Deterministic fallback: read the CTIG for this satellite's
        #    active THREAT_INDICATOR nodes and map subtype -> intent.
        for node, data in ctig.graph.nodes(data=True):
            if (data.get("node_type") == CognitiveThreatGraph.NODE_TYPE_THREAT
                    and data.get("origin_satellite") == satellite_id):
                return SUBTYPE_TO_INTENT.get(data.get("subtype"), "NOMINAL")

        return "NOMINAL"

    def infer_with_confidence(self, ctig: CognitiveThreatGraph, satellite_id: str,
                               state: Union[StateVector, dict]):
        state = coerce_state_vector(state)
        intent = self.infer(ctig, satellite_id, state)
        proba = self.registry.predict_proba(state) if self.registry.has_active() else None
        confidence = proba.get(intent, 1.0) if proba else 1.0
        return intent, confidence
