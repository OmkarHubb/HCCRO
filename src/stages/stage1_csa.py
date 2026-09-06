"""
Stage 1: Cyber Situation Awareness (CSA)
=========================================
Ingests raw telemetry from autonomous LEO satellites and maps it to a normalized
Operational State Vector S_t = {C, R, T, Q, M, E, A} based on Cognitive Cyber Resilience Theory (CCRT).

Includes abstract BaseTelemetryParser and DefaultTelemetryParser adapters for modular telemetry feeds.
"""

from abc import ABC, abstractmethod
import math
from typing import Dict, Any, Union, Optional
from src.core.interfaces import BaseStage, TelemetryData, StateVectorData
from src.core.state_vector import StateVector
from src.utils.logger import get_logger

logger = get_logger("Stage1.CSA")


class BaseTelemetryParser(ABC):
    """Abstract telemetry parser contract for plugging in live satellite telemetry decoders."""

    @abstractmethod
    def parse(self, raw_input: Dict[str, Any]) -> TelemetryData:
        pass


class DefaultTelemetryParser(BaseTelemetryParser):
    """Default telemetry parser with normalization and parameter aliasing fallbacks."""

    def parse(self, raw_input: Dict[str, Any]) -> TelemetryData:
        return TelemetryData.from_dict(raw_input)


class SlidingWindowTelemetryParser(BaseTelemetryParser):
    """
    MODULE 4 — Sliding temporal window telemetry parser.
    
    Maintains a circular buffer of the last W telemetry readings and computes
    rolling statistics (mean, variance, delta-change-rate) for key metrics:
    PDR, spectrum_usage, processor_utilization, navigation_consistency.
    
    These windowed features provide anomaly confidence scoring to the CSA
    normalization engine — a sudden spike in variance or delta-rate indicates
    a transient attack rather than a steady-state degradation.
    
    Paper Reference: Stage 1 CSA — "confidence and uncertainty associated
    with observed cyber events" requires temporal context beyond single-point
    readings.
    """

    # Tracked metrics for rolling statistics
    TRACKED_KEYS = [
        "packet_delivery_ratio", "spectrum_usage",
        "processor_utilization", "navigation_consistency",
    ]

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        # Circular buffer: list of dicts, newest at end
        self._buffer: list = []

    def parse(self, raw_input: Dict[str, Any]) -> TelemetryData:
        """Parses raw input AND updates the sliding window buffer."""
        telemetry = TelemetryData.from_dict(raw_input)
        # Store a snapshot of tracked metrics
        snapshot = {
            "packet_delivery_ratio": telemetry.packet_delivery_ratio,
            "spectrum_usage": telemetry.spectrum_usage,
            "processor_utilization": telemetry.processor_utilization,
            "navigation_consistency": telemetry.navigation_consistency,
        }
        self._buffer.append(snapshot)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)
        return telemetry

    # Alias for parse method
    push = parse

    def get_windowed_features(self) -> Dict[str, Dict[str, float]]:
        """
        Computes rolling mean, variance, and delta-change-rate for each
        tracked metric over the current window buffer.
        
        Returns:
            Dict keyed by metric name, each containing:
            - 'mean': rolling average over the window
            - 'variance': rolling variance (population) over the window
            - 'delta_rate': absolute change rate = |current - previous| / dt
            - 'window_fill': fraction of window buffer that is populated (0..1)
        """
        n = len(self._buffer)
        if n == 0:
            return {k: {"mean": 0.0, "variance": 0.0, "delta_rate": 0.0, "window_fill": 0.0}
                    for k in self.TRACKED_KEYS}

        features = {}
        fill_ratio = n / self.window_size

        for key in self.TRACKED_KEYS:
            values = [snap[key] for snap in self._buffer]
            mean_val = sum(values) / n
            var_val = sum((v - mean_val) ** 2 for v in values) / n
            # Delta-rate: absolute difference between last two readings
            if n >= 2:
                delta = abs(values[-1] - values[-2])
            else:
                delta = 0.0
            features[key] = {
                "mean": round(mean_val, 6),
                "variance": round(var_val, 6),
                "delta_rate": round(delta, 6),
                "window_fill": round(fill_ratio, 2),
            }
        return features

    def get_anomaly_confidence(self) -> float:
        """
        Computes a composite anomaly confidence score [0.0, 1.0] based on
        windowed statistics. High variance + high delta_rate in PDR or
        spectrum_usage increases confidence that an active attack is occurring.
        """
        feats = self.get_windowed_features()
        if feats["packet_delivery_ratio"]["window_fill"] < 0.3:
            return 0.5  # Insufficient data — neutral confidence

        # Weighted composite: variance spikes and delta-rate spikes indicate attacks
        pdr_var = feats["packet_delivery_ratio"]["variance"]
        spec_var = feats["spectrum_usage"]["variance"]
        cpu_delta = feats["processor_utilization"]["delta_rate"]
        nav_delta = feats["navigation_consistency"]["delta_rate"]

        # Normalize each component to [0,1] range with empirical scaling
        conf = min(1.0, (
            0.30 * min(1.0, pdr_var / 0.05) +       # PDR variance (nominal ~0.001)
            0.25 * min(1.0, spec_var / 0.05) +       # Spectrum variance
            0.25 * min(1.0, cpu_delta / 30.0) +      # CPU delta (nominal <2%)
            0.20 * min(1.0, nav_delta / 10.0)        # Nav drift delta (nominal <1m)
        ))
        return round(conf, 4)

    # Alias for get_anomaly_confidence
    compute_anomaly_confidence = get_anomaly_confidence

    def reset(self) -> None:
        """Clears the window buffer."""
        self._buffer.clear()


class CyberSituationAwareness(BaseStage):
    """
    Stage 1: Cyber Situation Awareness (CSA) Engine.
    Inherits from BaseStage to enforce plug-and-play adaptability.
    """

    def __init__(self, parser: Optional[BaseTelemetryParser] = None):
        super().__init__(name="Stage1_CSA")
        self.parser = parser or DefaultTelemetryParser()
        # MODULE 4: Optional sliding window for temporal enrichment
        self.sliding_window = SlidingWindowTelemetryParser()

    @staticmethod
    def _clip(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clips value strictly between min_val and max_val."""
        return max(min_val, min(max_val, float(value)))

    def _normalization_algorithm(self, telemetry: TelemetryData) -> StateVectorData:
        """
        Pluggable normalization math slot implementing standard CCRT S_t formulas:
        - Communication Integrity: C = PDR * (1.0 - spectrum_usage)
        - Routing Stability: R = 1.0 / (1.0 + (latency / 100.0))
        - Trust State: T = max(0.0, 1.0 - auth_events * 0.25) * e^(-0.05 * nav_drift)
        - Communication Quality: Q = PDR * (1.0 / (1.0 + (latency / 200.0)))
        - Resource Efficiency: E = energy_availability * (1.0 - max(CPU / 100.0, RAM / 100.0))
        - Autonomous Decision Reliability: A = T * software_integrity
        - Mission Performance: M = Q * E
        """
        cpu_frac = telemetry.processor_utilization / 100.0 if telemetry.processor_utilization > 1.0 else telemetry.processor_utilization
        ram_frac = telemetry.memory_consumption / 100.0 if telemetry.memory_consumption > 1.0 else telemetry.memory_consumption

        # 1. C (Communication Integrity)
        c_raw = telemetry.packet_delivery_ratio * (1.0 - telemetry.spectrum_usage)
        c_t = self._clip(c_raw)

        # 2. R (Routing Stability)
        r_raw = 1.0 / (1.0 + (telemetry.communication_latency / 100.0))
        r_t = self._clip(r_raw)

        # 3. T (Trust State)
        auth_trust = max(0.0, 1.0 - (telemetry.authentication_events * 0.25))
        drift_factor = math.exp(-0.05 * telemetry.navigation_consistency)
        t_t = self._clip(auth_trust * drift_factor)

        # 4. Q (Communication Quality)
        q_raw = telemetry.packet_delivery_ratio * (1.0 / (1.0 + (telemetry.communication_latency / 200.0)))
        q_t = self._clip(q_raw)

        # 5. E (Resource Efficiency)
        e_raw = telemetry.energy_availability * (1.0 - max(cpu_frac, ram_frac))
        e_t = self._clip(e_raw)

        # 6. A (Autonomous Decision Reliability)
        a_t = self._clip(t_t * telemetry.software_integrity_measurements)

        # 7. M (Mission Performance)
        m_t = self._clip(q_t * e_t)

        # Detect active threat indicators
        threats = []
        if c_t < 0.5 or telemetry.spectrum_usage > 0.5:
            threats.append("RF_JAMMING_SUSPECTED")
        if t_t < 0.5 or telemetry.navigation_consistency > 15.0:
            threats.append("GPS_SPOOFING_SUSPECTED")
        if e_t < 0.2 or cpu_frac > 0.85:
            threats.append("RESOURCE_DOS_SUSPECTED")

        return StateVectorData(
            C=round(c_t, 4),
            R=round(r_t, 4),
            T=round(t_t, 4),
            Q=round(q_t, 4),
            M=round(m_t, 4),
            E=round(e_t, 4),
            A=round(a_t, 4),
            timestamp=telemetry.timestamp,
            active_threats=threats,
        )

    def execute(self, input_data: Union[TelemetryData, Dict[str, Any]]) -> StateVectorData:
        """
        Executes Stage 1 processing using standardized interfaces.
        
        Args:
            input_data: TelemetryData object or raw dictionary.
            
        Returns:
            StateVectorData: Normalized S_t state vector.
        """
        if input_data is None:
            telemetry = TelemetryData()
        elif isinstance(input_data, dict):
            if "S_t" in input_data and isinstance(input_data["S_t"], StateVectorData):
                # Direct real-world GNSS parsed state vector
                return input_data["S_t"]
            telemetry = self.parser.parse(input_data)
            # MODULE 4: Also feed through sliding window for temporal context
            self.sliding_window.parse(input_data)
        else:
            telemetry = input_data

        logger.info("[Stage 1 CSA] Ingesting telemetry & computing S_t Operational State Vector.")
        return self._normalization_algorithm(telemetry)

    def process_telemetry(self, telemetry_data: Dict[str, Any]) -> Dict[str, float]:
        """Convenience method returning dictionary S_t vector."""
        state_data = self.execute(telemetry_data)
        return state_data.to_dict()

    def process(self, raw_telemetry: Dict[str, Any]) -> StateVector:
        """Integration bridge returning StateVector domain object."""
        state_data = self.execute(raw_telemetry)
        state_obj = StateVector(timestamp=state_data.timestamp)
        state_obj.update_from_telemetry(raw_telemetry)
        state_obj.s_t = state_data.to_dict()
        state_obj.active_threats = state_data.active_threats
        return state_obj


# Backward compatibility alias
Stage1CSA = CyberSituationAwareness


if __name__ == "__main__":
    csa = CyberSituationAwareness()
    print("=== HCCRO Stage 1 CSA (Abstract Base Stage Verified) ===")
    sample_raw = {
        "packet_delivery_ratio": 0.98,
        "communication_latency": 25.0,
        "spectrum_usage": 0.05,
        "processor_utilization": 12.0,
        "energy_availability": 0.95,
    }
    output = csa.execute(sample_raw)
    print("S_t State Vector Data:", output)
