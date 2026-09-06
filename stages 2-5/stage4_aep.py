"""
src/stages/stage4_aep.py

Stage 4 -- Attack Evolution Prediction (AEP)

Implements a probabilistic Independent Cascade Model (ICM) over the Stage 2
CTIG to forecast how an active threat laterally propagates to neighboring
satellites over a K-step lookahead horizon (t -> t+k), per spec section 3
(Stage 4).

Seeding
-------
Seed nodes are the satellites currently hosting an active THREAT_INDICATOR
in the CTIG (i.e. `ctig.active_threats()`), weighted by that threat's
severity.

Edge propagation probability
-----------------------------
For each directed `communication_link` edge (u -> v) with `weight` in
[0, 1] -- Stage 2's live link trust/quality, which `update_ctig()` actively
degrades on injection -- the per-tick activation probability is:

    p(u, v) = severity_u * (1 - weight(u, v)) * ICM_SCALE

i.e. a link Stage 2 has already weakened is *more* likely to carry the
attack onward, and a higher-severity origin threat raises the odds further
-- consistent with CCRT's assumption that a cyberattack is a perturbation
that propagates across interconnected operational dimensions.

Because a single ICM run is stochastic, `predict()` runs a Monte Carlo
ensemble and returns, for every satellite and every tick in the horizon,
the empirical probability that satellite becomes compromised by that tick
-- the adjacent-node infection risk map Stage 5 (MIA) consumes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

from src.stages.stage2_ctig import CognitiveThreatGraph

ICM_SCALE = 0.6  # global dampening factor on edge activation probability


@dataclass
class PropagationForecast:
    horizon_ticks: int
    # per-tick, per-satellite empirical probability of being compromised BY that tick
    compromise_prob_by_tick: List[Dict[str, float]] = field(default_factory=list)
    # satellites judged "likely compromised" (prob > threshold) by the final tick
    likely_compromised_final: List[str] = field(default_factory=list)

    def risk_map(self) -> Dict[str, float]:
        """Convenience accessor: final-tick per-satellite compromise probability."""
        return self.compromise_prob_by_tick[-1] if self.compromise_prob_by_tick else {}


class AttackEvolutionPredictor:
    """Stage 4: Monte-Carlo Independent Cascade Model for attack propagation."""

    def __init__(self, n_simulations: int = 500, compromise_threshold: float = 0.5,
                 seed: int = None) -> None:
        self.n_simulations = n_simulations
        self.compromise_threshold = compromise_threshold
        self._rng = random.Random(seed)

    def predict(self, ctig: CognitiveThreatGraph, horizon_ticks: int = 3) -> PropagationForecast:
        """K-step lookahead: `horizon_ticks` is K, the discrete number of
        future ticks (t+1 .. t+K) simulated."""
        satellites = ctig.satellites()
        seeds = self._seed_satellites(ctig)

        if not seeds:
            # nominal: nothing active, nothing propagates
            empty_tick = {sid: 0.0 for sid in satellites}
            return PropagationForecast(
                horizon_ticks=horizon_ticks,
                compromise_prob_by_tick=[dict(empty_tick) for _ in range(horizon_ticks)],
                likely_compromised_final=[],
            )

        # tick_counts[t][sat] = number of simulations where sat is active by tick t
        tick_counts = [dict.fromkeys(satellites, 0) for _ in range(horizon_ticks)]

        for _ in range(self.n_simulations):
            active = dict(seeds)  # satellite_id -> severity, currently active/newly-active
            ever_active = set(active.keys())

            for t in range(horizon_ticks):
                newly_active: Dict[str, float] = {}
                for u, severity in active.items():
                    for v in ctig.graph.successors(u):
                        if ctig.graph.nodes[v].get("node_type") != CognitiveThreatGraph.NODE_TYPE_SATELLITE:
                            continue
                        if v in ever_active:
                            continue
                        edge = ctig.graph.get_edge_data(u, v) or {}
                        if edge.get("relation") != CognitiveThreatGraph.EDGE_COMMUNICATION_LINK:
                            continue
                        link_weight = edge.get("weight", 0.5)  # current (possibly degraded) link quality
                        p = min(1.0, severity * (1.0 - link_weight) * ICM_SCALE)
                        if self._rng.random() < p:
                            newly_active[v] = max(severity * 0.85, 0.05)  # attack strength decays slightly per hop

                ever_active.update(newly_active.keys())
                for sid in ever_active:
                    tick_counts[t][sid] += 1

                active = newly_active
                if not active:
                    # cascade has died out; remaining ticks keep the same ever_active set
                    for remaining_t in range(t + 1, horizon_ticks):
                        for sid in ever_active:
                            tick_counts[remaining_t][sid] += 1
                    break

        compromise_prob_by_tick = [
            {sid: count / self.n_simulations for sid, count in tick.items()}
            for tick in tick_counts
        ]

        final_probs = compromise_prob_by_tick[-1]
        likely_compromised_final = [
            sid for sid, p in final_probs.items() if p >= self.compromise_threshold
        ]

        return PropagationForecast(
            horizon_ticks=horizon_ticks,
            compromise_prob_by_tick=compromise_prob_by_tick,
            likely_compromised_final=likely_compromised_final,
        )

    @staticmethod
    def _seed_satellites(ctig: CognitiveThreatGraph) -> Dict[str, float]:
        """Seeds = satellites currently hosting an active THREAT_INDICATOR,
        weighted by that threat's severity (max severity if multiple)."""
        seeds: Dict[str, float] = {}
        for node, data in ctig.graph.nodes(data=True):
            if data.get("node_type") == CognitiveThreatGraph.NODE_TYPE_THREAT:
                sat = data.get("origin_satellite")
                severity = data.get("severity", 0.5)
                if sat is not None:
                    seeds[sat] = max(seeds.get(sat, 0.0), severity)
        return seeds
