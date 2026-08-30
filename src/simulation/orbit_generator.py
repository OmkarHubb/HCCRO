"""
Orbit Dynamics Generator
========================
Simulates satellite orbital mechanics, position coordinates, and velocity vectors.
"""

import math
import time
from typing import Dict, Any

class OrbitGenerator:
    """Simulates orbit state (LEO/GEO parameters)."""

    def __init__(self, altitude_km: float = 550.0, inclination_deg: float = 97.5):
        self.altitude_km = altitude_km
        self.inclination_deg = inclination_deg

    def step_simulation(self, elapsed_seconds: float) -> Dict[str, Any]:
        """Calculates current satellite orbital coordinates."""
        omega = 2 * math.pi / (95.6 * 60)  # Orbital angular velocity
        lat = 15.0 * math.sin(omega * elapsed_seconds)
        lon = (20.0 * elapsed_seconds / 60.0) % 360.0 - 180.0
        
        return {
            "timestamp": time.time(),
            "altitude_km": self.altitude_km,
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "velocity_km_s": 7.66
        }
