"""
src/utils/state_vector.py

Shared dataclasses for the 7-D Operational State Vector (St) and the raw
SituationVector produced by Stage 1 (Cyber Situation Awareness).

These are intentionally lightweight so Stages 2-5 can be developed/tested
independently of your exact Stage 1 parser implementation. If your
`stage1_csa.py` already defines `StateVector` / `SituationVector`, just
import those instead -- the field names below match section 2.1 and 4.x
of the HCCRO v2 spec so the two should be drop-in compatible.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StateVector:
    """The 7-dimensional normalized Operational State Vector St = {C,R,T,Q,M,E,A}."""

    C: float = 1.0  # Communication Integrity
    R: float = 1.0  # Routing Stability
    T: float = 1.0  # Trust State
    Q: float = 1.0  # Communication Quality (PDR / bandwidth)
    M: float = 1.0  # Mission Performance
    E: float = 1.0  # Resource Efficiency (battery + CPU/mem margin)
    A: float = 1.0  # Autonomous Decision Reliability

    def clamp(self) -> "StateVector":
        for f in ("C", "R", "T", "Q", "M", "E", "A"):
            setattr(self, f, min(1.0, max(0.0, getattr(self, f))))
        return self

    def as_dict(self) -> dict:
        return {"C": self.C, "R": self.R, "T": self.T, "Q": self.Q,
                "M": self.M, "E": self.E, "A": self.A}

    @classmethod
    def from_dict(cls, d: dict) -> "StateVector":
        return cls(**{k: float(v) for k, v in d.items() if k in
                       ("C", "R", "T", "Q", "M", "E", "A")})


def coerce_state_vector(state) -> "StateVector":
    """Accepts either a StateVector or a plain dict {"C":..., "R":..., ...}
    (the Input Contract other stages/teammates rely on) and normalizes to
    a StateVector instance."""
    if isinstance(state, StateVector):
        return state
    if isinstance(state, dict):
        return StateVector.from_dict(state)
    raise TypeError(f"Expected StateVector or dict, got {type(state)}")


@dataclass
class SituationVector:
    """
    Raw (pre-normalization) situation snapshot for one satellite at one tick,
    fed by the Mendeley GNSS adapter and/or the OPS-SAT telemetry adapter
    (spec sections 4.1 / 4.2).
    """

    satellite_id: str

    # --- Mendeley GNSS (RF & navigation) fields ---
    cn0_db_hz: Optional[float] = None          # Carrier-to-noise density ratio
    spectrum_anomaly: float = 0.0              # 0-1 severity score
    lat_drift_m: float = 0.0
    lon_drift_m: float = 0.0
    alt_error_m: float = 0.0
    dop: Optional[float] = None                # Dilution of precision

    # --- ESA OPS-SAT (OS / DoS) fields ---
    cpu_load_pct: float = 0.0                  # SEPP CPU load, 0-100
    ram_usage_pct: float = 0.0                 # SEPP RAM usage, 0-100
    battery_pct: float = 100.0                 # Bus state of charge, 0-100
    network_delay_ms: float = 0.0

    # --- derived / cross-cutting ---
    trust_score: float = 1.0                   # peer-rating aggregate, 0-1
    pdr: float = 1.0                           # packet delivery ratio, 0-1

    extra: dict = field(default_factory=dict)
