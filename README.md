# Hierarchical Cognitive Cyber Resilience Optimization (HCCRO)

Production-grade satellite security framework for space asset cyber resilience, threat injection simulation, and automated multi-objective response optimization.

## 🛰️ Framework Architecture

The framework is organized into 8 cognitive resilience stages:

1. **Stage 1 (CSA):** Cyber Situation Awareness (Telemetry mapping to $S_t$)
2. **Stage 2 (CTIG):** Cognitive Threat Intelligence Graph (Dependency modeling)
3. **Stage 3 (AIM):** Attack Intention Modeling (Inference engine)
4. **Stage 4 (AEP):** Attack Evolution Prediction (Lateral movement)
5. **Stage 5 (MIA):** Mission Impact Estimation (Operational risk & MCI)
6. **Stage 6 (OPT):** Hierarchical Multi-Objective Optimization (Countermeasure selection)
7. **Stage 7 (HEAL):** Distributed Self-Healing & PACE Fallbacks (Execution)
8. **Stage 8 (CKE):** Cyber Knowledge Evolution (Database logging & metric tracking)

## 📁 Directory Layout

```text
hccro_resilience_project/
├── config/              # Configuration parameters (orbits, thresholds, solver weights)
├── data/                # Telemetry datasets (raw/processed)
├── src/
│   ├── core/            # Main orchestrator & S_t state vector representation
│   ├── stages/          # Decoupled modules for Stages 1 to 8
│   ├── simulation/      # Orbit kinematics & attack injection environment
│   └── utils/           # Colorized logging & metrics (MCI, CRI, DREI)
└── tests/               # Automated pytest suite
```

## 🚀 Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run tests:
   ```bash
   pytest tests/
   ```

3. Run pipeline demonstration:
   ```bash
   python -m src.core.orchestrator
   ```
