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


class CyberSituationAwareness(BaseStage):
    """
    Stage 1: Cyber Situation Awareness (CSA) Engine.
    Inherits from BaseStage to enforce plug-and-play adaptability.
    """

    def __init__(self, parser: Optional[BaseTelemetryParser] = None):
        super().__init__(name="Stage1_CSA")
        self.parser = parser or DefaultTelemetryParser()

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
        if isinstance(input_data, dict):
            telemetry = self.parser.parse(input_data)
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
