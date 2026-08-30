"""
Continuous Telemetry Streamer Generator
=======================================
Combines orbit kinematics generator with attack injector to yield data stream.
"""

import time
from typing import Generator, Dict, Any
from src.simulation.orbit_generator import OrbitGenerator
from src.simulation.attack_injector import AttackInjector

class TelemetryStreamer:
    """Streams continuous telemetry frames for test and execution."""

    def __init__(self):
        self.orbit_sim = OrbitGenerator()
        self.injector = AttackInjector()

    def generate_stream(self, duration_sec: int = 10, interval_sec: float = 1.0) -> Generator[Dict[str, Any], None, None]:
        """Yields continuous telemetry frames."""
        start_time = time.time()
        for i in range(duration_sec):
            elapsed = time.time() - start_time
            frame = self.orbit_sim.step_simulation(elapsed)
            frame["snr_db"] = 24.5
            frame["pseudorange_error_m"] = 1.2
            frame["cpu_usage_pct"] = 18.0

            # Inject simulated attack frame at tick 3
            if i == 3:
                frame = self.injector.inject_rf_jamming(frame)
            elif i == 6:
                frame = self.injector.inject_gps_spoofing(frame)

            yield frame
            time.sleep(interval_sec)
