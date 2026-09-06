# HCCRO Design Specification & System Architecture (Production v3.0)

## Executive Summary
The **Hierarchical Cognitive Cyber Resilience Optimization (HCCRO)** framework provides continuous, autonomous cyber resilience for satellite constellations operating in contested space environments. The system processes multi-source space telemetry, builds real-time threat graphs, infers multi-step adversarial intent, optimizes onboard power/compute resource allocation under non-linear constraints, and executes distributed self-healing countermeasures.

---

## System Architecture

```
                                +----------------------------------+
                                |  Layer 4: Ground Station Node    |
                                |  (Out-of-band Retraining Root)   |
                                +----------------------------------+
                                                 |
                                                 v
                                +----------------------------------+
                                | Layer 3: Constellation Manager   |
                                | (Orbit-wide PACE Escalation)     |
                                +----------------------------------+
                                                 |
                                                 v
                                +----------------------------------+
                                |  Layer 2: Cluster Proxy Nodes    |
                                |  (Consensus & Workload Bidding)  |
                                +----------------------------------+
                                                 |
                                                 v
                                +----------------------------------+
                                |   Layer 1: Satellite Nodes       |
                                |   (Local 8-Stage Cognitive Loop) |
                                +----------------------------------+
```

---

## 8-Stage Cognitive Pipeline Specification

1. **Stage 1: Cyber Situation Awareness (CSA)**
   - Ingests raw telemetry feeds (e.g., C/N0, pseudorange, ECEF coordinates, gDOP, CPU load).
   - Maps parameters to normalized 7-D State Vector $S_t = \{C, R, T, Q, M, E, A\}$.
   - Evaluates sliding temporal window ($W=10$) for rolling variance and anomaly confidence.

2. **Stage 2: Cognitive Threat Intelligence Graph (CTIG)**
   - Constructs dynamic NetworkX topology graph.
   - Calculates PageRank centrality scores to identify critical node vulnerabilities.
   - Traces critical attack corridors via Dijkstra shortest path algorithm.

3. **Stage 3: Attack Intention Modeling (AIM)**
   - Executes Bayesian Intent Inference Engine for multi-step attack chains ($P(\text{Jamming} \mid \text{Low C/N0})$, $P(\text{Spoofing} \mid \text{High Drift, Low SV Ratio})$, $P(\text{DoS} \mid \text{Elevated CPU})$).
   - Combines Bayesian posterior, graph heuristic, and registered ML models in a 3-source weighted ensemble.

4. **Stage 4: Attack Evolution Prediction (AEP)**
   - Models lateral threat propagation across constellation nodes using the Independent Cascade Model (ICM).

5. **Stage 5: Mission Impact Estimation (MIA)**
   - Computes Mission Degradation Index (MDI) and Mission Continuity Index (MCI = $M_t \times R_t$).

6. **Stage 6: Hierarchical Multi-Objective Optimization & PACE Decision Brain**
   - DCS-MOS Dynamic Weight Controller: Suppresses comm weights ($\delta$) and spikes healing weights ($\lambda$) under low battery ($E_t < 0.30$) or low trust ($T_t < 0.30$).
   - SciPy SLSQP Solver (`scipy.optimize.minimize`): Solves $U(x) = \alpha R + \beta M + \gamma T + \delta C + \lambda H$ subject to $E_{\text{draw}}(x) \le E_{\text{avail}}$ and $L(x) \le L_{\text{max}}$.
   - DRC-DREI PACE Solver: Transitions between Primary, Alternate, Contingency, and Emergency modes. Applies strategic recovery multiplier $\kappa=1.2$ for upward recovery and locks exploration rate $\epsilon=0.0$ strictly in Emergency mode.

7. **Stage 7: Distributed Self-Healing Execution (DSH)**
   - Node Isolation: Drops communication edges connected to compromised nodes ($T_t < 0.30$).
   - Dijkstra Rerouting: Computes alternative secure pathways around isolated nodes.
   - Assistance Feasibility Score (AFS) Bidding: Migrates heavy workload (`SensorDataProcessing`) to optimal peers when local CPU load $>70\%$.

8. **Stage 8: Cyber Knowledge Evolution (CKE)**
   - SQLite Database (`cke_database.db`): Logs incident cycles, mitigation outcomes, and policy weights.
   - Queries historical success rates to adaptively modify PACE state transition probabilities and Stage 6 optimization weights.
