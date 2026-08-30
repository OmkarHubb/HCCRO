"""
Threat Attack Injector Module
=============================
Simulates RF Jamming, GPS Spoofing, and Resource DoS attack vectors.
"""

from typing import Dict, Any

class AttackInjector:
    """Injects synthetic security threats into telemetry feeds."""

    def __init__(self):
        self.jamming_active = False
        self.spoofing_active = False
        self.dos_active = False

    def inject_rf_jamming(self, telemetry: Dict[str, Any], drop_db: float = 18.0) -> Dict[str, Any]:
        """Simulates RF jammer lowering Signal-to-Noise Ratio (SNR)."""
        telemetry["snr_db"] = max(0.0, telemetry.get("snr_db", 25.0) - drop_db)
        return telemetry

    def inject_gps_spoofing(self, telemetry: Dict[str, Any], bias_meters: float = 50.0) -> Dict[str, Any]:
        """Simulates GPS spoofing introducing pseudorange error."""
        telemetry["pseudorange_error_m"] += bias_meters
        telemetry["gps_lock"] = False
        return telemetry

    def inject_resource_dos(self, telemetry: Dict[str, Any], cpu_load: float = 98.5) -> Dict[str, Any]:
        """Simulates onboard processor resource exhaustion."""
        telemetry["cpu_usage_pct"] = cpu_load
        return telemetry
