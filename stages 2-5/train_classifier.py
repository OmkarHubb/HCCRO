"""
src/utils/train_classifier.py

Task 6 -- Train and Plug in the ML Threat Classifier.

Trains a lightweight classifier (RandomForest by default; SVM/MLP available
via --model) whose input is the 7-D Operational State Vector
St = {C, R, T, Q, M, E, A} -- the same object Stage 6 and the Stage 8
SQLite log consume -- and whose output is one of the four standardized
Stage 3 intent strings.

Dataset provenance (corrected)
-------------------------------
- **Jamming / Spoofing** training rows are derived from the columns the
  **Mendeley GNSS dataset** actually contains: Carrier-to-Noise density
  ratio (C/N0), Dilution of Precision (DOP), and lat/lon/alt coordinate
  drift. `mendeley_row_to_state_vector()` maps those raw readings onto St
  using the same relationships Stage 2 uses for its anomaly thresholds
  (C/N0 collapse -> communication integrity & quality drop; drift/DOP
  spike -> trust & autonomous-decision-reliability drop).
- **DoS** training rows come from a **simulated host resource-saturation
  curve** -- CPU utilization climbing exponentially toward 100% together
  with a buffer/queue fill curve consistent with socket-flooding behavior
  -- via `simulate_dos_state_vector()`. This is intentional: the real ESA
  **OPS-SAT-AD** dataset contains only 9 physical channels (3 magnetometer
  + 6 photodiode/sun-sensor readings; see Ruszczak et al., Scientific Data,
  2024/2025) and has no CPU-load, RAM, or DoS-labeled columns at all, so it
  is **not** used here as a source of "sepp_cpu_load" / "sepp_ram_used"
  features -- those columns do not exist in OPS-SAT-AD. If you want a real
  (rather than simulated) host-based DoS signal, swap in a proper IoT
  host-telemetry dataset such as Edge-IIoTset, mapped onto St the same way.
  OPS-SAT-AD's actual channels are a reasonable fit for Stage 1's generic
  nominal-vs-anomaly *pre-filter*, not for Stage 3's attack-intent labels,
  since nothing in those 9 channels is attack-type-specific.

CLI
---
    python -m src.utils.train_classifier --model rf
    python -m src.utils.train_classifier --model svm
    python -m src.utils.train_classifier --model mlp
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.stages.stage3_aim import FEATURE_NAMES, PredictiveModelRegistry, extract_features
from src.utils.state_vector import StateVector

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


# ---------------------------------------------------------------------------
# Raw telemetry -> 7-D State Vector mappings
# ---------------------------------------------------------------------------
def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def mendeley_row_to_state_vector(cn0_db_hz: float, dop: float, gps_drift_m: float) -> StateVector:
    """Maps one Mendeley-GNSS-style reading (C/N0, DOP, coordinate drift)
    onto St. Nominal C/N0 ~= 40-50 dB-Hz; jamming drives it down toward
    5-20 dB-Hz. Nominal drift is sub-meter; spoofing drives it to tens of
    meters, usually accompanied by a DOP spike."""
    comm = _clamp01((cn0_db_hz - 5.0) / 40.0)
    trust = _clamp01(1.0 - gps_drift_m / 60.0 - max(0.0, dop - 1.5) / 8.0)
    autonomy = _clamp01(1.0 - gps_drift_m / 50.0 - max(0.0, dop - 1.5) / 6.0)
    routing = _clamp01(1.0 - 0.3 * min(1.0, gps_drift_m / 100.0))
    mission = _clamp01(1.0 - 0.5 * (1.0 - comm))
    return StateVector(C=comm, R=routing, T=trust, Q=comm, M=mission, E=0.9, A=autonomy).clamp()


def simulate_dos_resource_saturation(n_samples: int, rng: np.random.Generator):
    """Simulated host resource-saturation curve: CPU utilization approaches
    100% exponentially over an attack's duration, with a buffer/queue-fill
    curve consistent with socket-flooding, both sampled at random points
    along the curve (plus noise) to cover the full severity range."""
    t = rng.uniform(0.0, 1.0, size=n_samples)
    cpu_util = 100.0 * (1.0 - np.exp(-5.0 * t)) + rng.normal(0, 3.0, size=n_samples)
    buffer_fill = 100.0 * (1.0 - np.exp(-4.0 * t)) + rng.normal(0, 3.0, size=n_samples)
    return np.clip(cpu_util, 0, 100), np.clip(buffer_fill, 0, 100)


def dos_reading_to_state_vector(cpu_util_pct: float, buffer_fill_pct: float) -> StateVector:
    load = max(cpu_util_pct, buffer_fill_pct) / 100.0
    efficiency = _clamp01(1.0 - load)
    mission = _clamp01(1.0 - 0.7 * load)
    routing = _clamp01(1.0 - 0.5 * load)
    autonomy = _clamp01(1.0 - 0.4 * load)
    quality = _clamp01(1.0 - 0.3 * load)
    return StateVector(C=0.85, R=routing, T=0.85, Q=quality, M=mission, E=efficiency, A=autonomy).clamp()


def nominal_state_vector(rng: np.random.Generator) -> StateVector:
    vals = np.clip(rng.normal(0.95, 0.03, size=7), 0.0, 1.0)
    return StateVector(*vals).clamp()


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def make_dataset(n_per_class: int = 600, seed: int = 42):
    rng = np.random.default_rng(seed)
    rows, labels = [], []

    # NOMINAL
    for _ in range(n_per_class):
        rows.append(extract_features(nominal_state_vector(rng)))
    labels += ["NOMINAL"] * n_per_class

    # RF_JAMMING_DISRUPTION -- Mendeley GNSS, C/N0 collapse dominant
    cn0 = rng.normal(13.0, 4.0, size=n_per_class)
    dop = rng.normal(1.6, 0.4, size=n_per_class)
    drift = rng.normal(1.0, 0.5, size=n_per_class)
    for c, d, g in zip(cn0, dop, drift):
        rows.append(extract_features(mendeley_row_to_state_vector(c, d, max(0.0, g))))
    labels += ["RF_JAMMING_DISRUPTION"] * n_per_class

    # COMMAND_SPOOFING_TAKEOVER -- Mendeley GNSS, drift/DOP spike dominant
    cn0 = rng.normal(36.0, 4.0, size=n_per_class)
    dop = rng.normal(6.0, 1.5, size=n_per_class)
    drift = rng.normal(40.0, 12.0, size=n_per_class)
    for c, d, g in zip(cn0, dop, drift):
        rows.append(extract_features(mendeley_row_to_state_vector(c, d, max(0.0, g))))
    labels += ["COMMAND_SPOOFING_TAKEOVER"] * n_per_class

    # RESOURCE_DENIAL_OF_SERVICE -- simulated resource-saturation curve
    cpu, buf = simulate_dos_resource_saturation(n_per_class, rng)
    for cu, bf in zip(cpu, buf):
        rows.append(extract_features(dos_reading_to_state_vector(cu, bf)))
    labels += ["RESOURCE_DENIAL_OF_SERVICE"] * n_per_class

    return np.vstack(rows), np.array(labels)


def build_model(kind: str):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    if kind == "svm":
        return SVC(kernel="rbf", probability=True, random_state=42)
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
    raise ValueError(f"Unknown model kind: {kind}")


class _ScaledModel:
    """Bundles a fitted StandardScaler + classifier behind the .predict /
    .predict_proba interface PredictiveModelRegistry expects."""

    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model = model
        self.classes_ = model.classes_

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))


def train_and_register(registry: PredictiveModelRegistry, kind: str = "rf",
                        X: np.ndarray = None, y: np.ndarray = None,
                        save: bool = True) -> dict:
    if X is None or y is None:
        X, y = make_dataset()

    label_encoder = LabelEncoder().fit(y)
    y_enc = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    scaler = StandardScaler().fit(X_train)
    model = build_model(kind)
    model.fit(scaler.transform(X_train), y_train)

    y_pred = model.predict(scaler.transform(X_test))
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

    pipeline = _ScaledModel(scaler, model)
    registry.register(f"threat_classifier_{kind}", pipeline, label_encoder,
                       feature_fn=extract_features, make_active=True)

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump({"pipeline": pipeline, "label_encoder": label_encoder, "kind": kind},
                    os.path.join(MODELS_DIR, f"threat_classifier_{kind}.joblib"))

    return {"report": report, "kind": kind}


def load_registered_model(registry: PredictiveModelRegistry, kind: str = "rf") -> None:
    """Loads a previously-saved model from disk and registers it."""
    path = os.path.join(MODELS_DIR, f"threat_classifier_{kind}.joblib")
    bundle = joblib.load(path)
    registry.register(f"threat_classifier_{kind}", bundle["pipeline"],
                       bundle["label_encoder"], feature_fn=extract_features, make_active=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "svm", "mlp"], default="rf")
    args = parser.parse_args()

    reg = PredictiveModelRegistry()
    result = train_and_register(reg, kind=args.model)
    print(f"Trained {args.model} classifier on St=[{', '.join(FEATURE_NAMES)}] -- "
          f"accuracy: {result['report']['accuracy']:.3f}")
    for label in ["NOMINAL", "RF_JAMMING_DISRUPTION", "COMMAND_SPOOFING_TAKEOVER",
                  "RESOURCE_DENIAL_OF_SERVICE"]:
        m = result["report"][label]
        print(f"  {label:28s} precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1-score']:.3f}")
