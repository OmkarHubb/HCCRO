"""
src/stages/stage5_mia.py

Stage 5 -- Mission Impact Estimation (MIA)

Consumes Stage 4's propagation forecast (per-satellite compromise
probability across the K-step horizon) and Stage 2's live CTIG, and
translates them into the operational consequences the spec calls for
(section 3, Stage 5):

  - expected drop in Packet Delivery Ratio (PDR) per satellite
  - expected drop in available bandwidth / communication quality per satellite
  - **expected degradation of routing utility on every communication_link
    edge** whose endpoints are forecast to have elevated compromise risk
  - the dynamic Mission Degradation Index (MDI) for the constellation
  - risk-adjusted state-utility weights omega*_t(s), which feed straight
    into Stage 6's DREI calculation (spec eq. 8): higher compromise
    probability pulls a satellite's effective weight toward the
    self-healing/recovery priority (kappa = 1.2, per spec section 2.4)

`MissionImpactReport.to_dict()` returns the formatted risk-map dictionary
`stage6_optimization.py` is expected to parse directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Union

from src.stages.stage2_ctig import CognitiveThreatGraph
from src.stages.stage4_aep import PropagationForecast
from src.utils.state_vector import StateVector, coerce_state_vector

# Recovery multiplier from spec section 2.4 (DREI, eq. 8)
KAPPA_RECOVERY = 1.2

# Degradation floors: how far PDR / comms quality can realistically fall
# under a fully-realized attack on a single satellite.
PDR_FLOOR = 0.25
BANDWIDTH_FLOOR = 0.30

# How strongly elevated endpoint compromise risk degrades a link's expected
# routing utility (0 = no effect, 1 = fully collapses to zero utility).
LINK_RISK_IMPACT_FACTOR = 0.8


@dataclass
class SatelliteImpact:
    satellite_id: str
    compromise_probability: float
    expected_pdr_drop: float
    expected_bandwidth_drop: float
    omega_star: float  # risk-adjusted utility weight for Stage 6

    def to_dict(self) -> dict:
        return {
            "compromise_probability": self.compromise_probability,
            "expected_pdr_drop": self.expected_pdr_drop,
            "expected_bandwidth_drop": self.expected_bandwidth_drop,
            "omega_star": self.omega_star,
        }


@dataclass
class LinkImpact:
    endpoints: Tuple[str, str]
    base_weight: float
    expected_weight: float
    routing_utility_penalty: float

    def to_dict(self) -> dict:
        return {
            "base_weight": self.base_weight,
            "expected_weight": self.expected_weight,
            "routing_utility_penalty": self.routing_utility_penalty,
        }


@dataclass
class MissionImpactReport:
    per_satellite: Dict[str, SatelliteImpact] = field(default_factory=dict)
    links: Dict[Tuple[str, str], LinkImpact] = field(default_factory=dict)
    mission_degradation_index: float = 0.0  # MDI, 0 (no impact) - 1 (total loss)

    def to_dict(self) -> dict:
        """Formatted risk-map dictionary ready for `stage6_optimization.py`."""
        return {
            "per_satellite": {sid: impact.to_dict() for sid, impact in self.per_satellite.items()},
            "links": {f"{u}->{v}": impact.to_dict() for (u, v), impact in self.links.items()},
            "mission_degradation_index": self.mission_degradation_index,
        }


class MissionImpactEstimator:
    """Stage 5: forecast + CTIG -> quantitative mission consequences."""

    def assess(self, ctig: CognitiveThreatGraph, forecast: PropagationForecast,
               current_states: Dict[str, Union[StateVector, dict]]) -> MissionImpactReport:
        states = {sid: coerce_state_vector(s) for sid, s in current_states.items()}
        final_probs = forecast.risk_map()

        per_satellite: Dict[str, SatelliteImpact] = {}
        weighted_degradation = 0.0
        total_mission_weight = 0.0

        for sat_id, state in states.items():
            p = final_probs.get(sat_id, 0.0)

            expected_pdr_drop = max(0.0, p * (state.Q - PDR_FLOOR)) if state.Q > PDR_FLOOR else 0.0
            expected_bw_drop = max(0.0, p * (state.C - BANDWIDTH_FLOOR)) if state.C > BANDWIDTH_FLOOR else 0.0

            # omega*: baseline mission-performance weight, boosted toward
            # recovery priority (kappa) in proportion to compromise risk --
            # this is the quantity Stage 6 plugs into the DREI numerator.
            omega_star = state.M * (1.0 + (KAPPA_RECOVERY - 1.0) * p)

            per_satellite[sat_id] = SatelliteImpact(
                satellite_id=sat_id,
                compromise_probability=p,
                expected_pdr_drop=expected_pdr_drop,
                expected_bandwidth_drop=expected_bw_drop,
                omega_star=omega_star,
            )

            weighted_degradation += p * state.M
            total_mission_weight += state.M

        mdi = weighted_degradation / total_mission_weight if total_mission_weight > 0 else 0.0

        # --- Link Degradation Mapping ---------------------------------
        # For every communication_link edge, degrade its expected routing
        # utility in proportion to the higher of its two endpoints'
        # compromise probabilities.
        links: Dict[Tuple[str, str], LinkImpact] = {}
        for u, v, data in ctig.communication_links():
            base_weight = data.get("weight", 1.0)
            risk = max(final_probs.get(u, 0.0), final_probs.get(v, 0.0))
            expected_weight = base_weight * (1.0 - LINK_RISK_IMPACT_FACTOR * risk)
            links[(u, v)] = LinkImpact(
                endpoints=(u, v),
                base_weight=base_weight,
                expected_weight=max(0.0, expected_weight),
                routing_utility_penalty=max(0.0, base_weight - expected_weight),
            )

        return MissionImpactReport(per_satellite=per_satellite, links=links,
                                    mission_degradation_index=mdi)
