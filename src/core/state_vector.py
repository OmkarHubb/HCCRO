"""
Operational State Vector (S_t) Module
=====================================
Represents the comprehensive operational, physical, and cyber state vector (S_t = {C, R, T, Q, M, E, A})
of the satellite payload and bus system at timestamp t, including the active PACE onboard sub-layer state.
"""

import time
from typing import Dict, Any, Optional, List
from src.core.interfaces import PaceState, StateVectorData


class StateVector:
    """
    Encapsulates the state vector S_t = <Physical, Cyber, Threat, Operational, PACE State>.
    """

    def __init__(self, timestamp: Optional[float] = None):
        self.timestamp: float = timestamp or time.time()
        self.altitude_km: float = 550.0
        self.snr_db: float = 25.0
        self.gps_lock: bool = True
        self.pseudorange_error_m: float = 0.0
        self.cpu_usage_pct: float = 20.0
        self.memory_usage_pct: float = 35.0
        self.power_level_pct: float = 98.0
        self.pace_state: PaceState = PaceState.PRIMARY
        self.s_t: Dict[str, float] = {
            "C": 1.0,
            "R": 1.0,
            "T": 1.0,
            "Q": 1.0,
            "M": 1.0,
            "E": 1.0,
            "A": 1.0,
        }
        self.active_threats: List[str] = []

    def update_from_telemetry(self, raw_data: Dict[str, Any]) -> None:
        """Updates internal state vector attributes from raw telemetry dictionary."""
        self.timestamp = raw_data.get("timestamp", self.timestamp)
        self.altitude_km = raw_data.get("altitude_km", self.altitude_km)
        self.snr_db = raw_data.get("snr_db", self.snr_db)
        self.gps_lock = raw_data.get("gps_lock", self.gps_lock)
        self.pseudorange_error_m = raw_data.get("pseudorange_error_m", self.pseudorange_error_m)
        self.cpu_usage_pct = raw_data.get("cpu_usage_pct", self.cpu_usage_pct)
        self.memory_usage_pct = raw_data.get("memory_usage_pct", self.memory_usage_pct)
        self.power_level_pct = raw_data.get("power_level_pct", self.power_level_pct)

    def to_state_vector_data(self) -> StateVectorData:
        """Converts to strongly-typed StateVectorData dataclass."""
        return StateVectorData(
            C=self.s_t.get("C", 1.0),
            R=self.s_t.get("R", 1.0),
            T=self.s_t.get("T", 1.0),
            Q=self.s_t.get("Q", 1.0),
            M=self.s_t.get("M", 1.0),
            E=self.s_t.get("E", 1.0),
            A=self.s_t.get("A", 1.0),
            pace_state=self.pace_state,
            timestamp=self.timestamp,
            active_threats=list(self.active_threats),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes state vector to dictionary format."""
        return {
            "timestamp": self.timestamp,
            "altitude_km": self.altitude_km,
            "snr_db": self.snr_db,
            "gps_lock": self.gps_lock,
            "pseudorange_error_m": self.pseudorange_error_m,
            "cpu_usage_pct": self.cpu_usage_pct,
            "memory_usage_pct": self.memory_usage_pct,
            "power_level_pct": self.power_level_pct,
            "pace_state": self.pace_state.value if isinstance(self.pace_state, PaceState) else str(self.pace_state),
            "s_t": dict(self.s_t),
            "active_threats": list(self.active_threats),
        }
