"""
Configuration Parameters for HCCRO Framework
============================================
Contains system operational parameters, orbit dynamics constants, anomaly detection thresholds,
PACE state thresholds, and optimization weightings for Stage 6 multi-objective solver.
"""

from pydantic import BaseModel
from typing import Dict, Any

class SystemConfig(BaseModel):
    # Orbit Parameters
    DEFAULT_ALTITUDE_KM: float = 550.0
    DEFAULT_INCLINATION_DEG: float = 97.5
    ORBIT_PERIOD_MINUTES: float = 95.6

    # Telemetry & Sampling
    TELEMETRY_SAMPLING_RATE_HZ: float = 10.0
    WINDOW_SIZE_SECONDS: int = 60

    # Threat Detection Thresholds
    RF_SNR_THRESHOLD_DB: float = 12.0
    GPS_PSEUDORANGE_BIAS_LIMIT_M: float = 15.0
    CPU_UTILIZATION_ALERT_PCT: float = 85.0
    MEMORY_UTILIZATION_ALERT_PCT: float = 90.0

    # Objective Function Weights (Stage 6)
    # Maximize U = [alpha*R + beta*M + gamma*T + delta*C + lambda*H]
    WEIGHT_R_RESILIENCE: float = 0.25      # alpha
    WEIGHT_M_MISSION: float = 0.25         # beta
    WEIGHT_T_TRUST: float = 0.20           # gamma
    WEIGHT_C_COMMUNICATION: float = 0.15   # delta
    WEIGHT_H_HEALING: float = 0.15         # lambda

    # Adaptive Weight Controller (Hardware-Saturating Feedback Loop)
    BATTERY_CRITICAL_THRESHOLD: float = 0.40  # 40% battery charge
    HEALING_SPIKE_VALUE: float = 100.0        # Exponential spike for lambda when battery is critical

    # Dynamic Redundancy Efficiency Index (DREI) Parameters
    STRATEGIC_RECOVERY_MULTIPLIER_KAPPA: float = 1.2

settings = SystemConfig()
