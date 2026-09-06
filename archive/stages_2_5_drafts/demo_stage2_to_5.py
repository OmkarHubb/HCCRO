"""
src/core/demo_stage2_to_5.py

End-to-end sanity check wiring Stages 2 -> 3 -> 4 -> 5 together over the
standard 5-satellite cycle topology (SC1..SC5), run for four scenarios:
nominal, jamming, spoofing, and a DoS resource-saturation spike. Run with:

    python -m src.core.demo_stage2_to_5
"""

from src.stages.stage2_ctig import CognitiveThreatGraph
from src.stages.stage3_aim import AttackIntentionModeler, PredictiveModelRegistry
from src.stages.stage4_aep import AttackEvolutionPredictor
from src.stages.stage5_mia import MissionImpactEstimator
from src.utils.state_vector import StateVector
from src.utils.train_classifier import (dos_reading_to_state_vector,
                                         mendeley_row_to_state_vector,
                                         nominal_state_vector, train_and_register)
import numpy as np

SAT_IDS = [f"SC{i}" for i in range(1, 6)]


def run_scenario(name: str, telemetry_by_sat: dict, state_by_sat: dict,
                  modeler: AttackIntentionModeler):
    print(f"\n=== Scenario: {name} ===")
    ctig = CognitiveThreatGraph.build_cycle_topology(SAT_IDS)
    states = {sid: StateVector() for sid in SAT_IDS}
    states.update(state_by_sat)

    for sid, telemetry in telemetry_by_sat.items():
        events = ctig.update_ctig(sid, telemetry, states[sid])
        for e in events:
            intent, confidence = modeler.infer_with_confidence(ctig, sid, states[sid])
            print(f"  [{sid}] injected {e.subtype} threat (severity={e.severity:.2f}), "
                  f"degraded {len(e.degraded_links)} link(s) -> inferred intent: "
                  f"{intent} (confidence={confidence:.2f})")

    predictor = AttackEvolutionPredictor(n_simulations=300, seed=7)
    forecast = predictor.predict(ctig, horizon_ticks=3)
    print("  Propagation forecast (P[compromised] by final tick):")
    for sid, p in forecast.risk_map().items():
        print(f"    {sid}: {p:.2f}")

    estimator = MissionImpactEstimator()
    report = estimator.assess(ctig, forecast, states)
    print(f"  Mission Degradation Index (MDI): {report.mission_degradation_index:.3f}")
    for sid, impact in report.per_satellite.items():
        if impact.compromise_probability > 0:
            print(f"    {sid}: PDR drop={impact.expected_pdr_drop:.3f}, "
                  f"bandwidth drop={impact.expected_bandwidth_drop:.3f}, "
                  f"omega*={impact.omega_star:.3f}")
    degraded_links = {k: v for k, v in report.links.items() if v.routing_utility_penalty > 0.01}
    if degraded_links:
        print("  Degraded communication_link edges:")
        for (u, v), impact in degraded_links.items():
            print(f"    {u}->{v}: weight {impact.base_weight:.2f} -> "
                  f"{impact.expected_weight:.2f} (penalty {impact.routing_utility_penalty:.2f})")


def main():
    registry = PredictiveModelRegistry()
    result = train_and_register(registry, kind="rf", save=False)
    print(f"Stage 3 classifier trained on St=[C,R,T,Q,M,E,A] -- "
          f"accuracy: {result['report']['accuracy']:.3f}")
    modeler = AttackIntentionModeler(registry)
    rng = np.random.default_rng(0)

    run_scenario("Nominal", {}, {sid: nominal_state_vector(rng) for sid in SAT_IDS}, modeler)

    jam_state = mendeley_row_to_state_vector(cn0_db_hz=12.0, dop=1.6, gps_drift_m=0.5)
    run_scenario("Jamming on SC2",
                 {"SC2": {"cn0_db_hz": 12.0}},
                 {"SC2": jam_state}, modeler)

    spoof_state = mendeley_row_to_state_vector(cn0_db_hz=36.0, dop=6.5, gps_drift_m=45.0)
    run_scenario("Spoofing on SC3",
                 {"SC3": {"lat_drift_m": 25.0, "lon_drift_m": 20.0, "alt_error_m": 10.0}},
                 {"SC3": spoof_state}, modeler)

    dos_state = dos_reading_to_state_vector(cpu_util_pct=97.0, buffer_fill_pct=94.0)
    run_scenario("DoS on SC4",
                 {"SC4": {"cpu_util_pct": 97.0, "buffer_fill_pct": 94.0}},
                 {"SC4": dos_state}, modeler)


if __name__ == "__main__":
    main()
