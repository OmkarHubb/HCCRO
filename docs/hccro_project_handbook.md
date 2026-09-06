# HCCRO Project Handbook & Developer Onboarding Guide

Welcome to the **HCCRO (Hierarchical Cognitive Cyber Resilience Optimization)** codebase! This handbook guides researchers, software engineers, and DevOps leads through system setup, execution, validation, and benchmarking.

---

## Quick Start Guide

### 1. Installation & Environment Setup
Ensure Python 3.9+ is installed. Install required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running Master Integrated System Benchmark
Execute the master integrated system pipeline across all 3,600 temporal epochs of the December 21, 2023 Mendeley GNSS dataset:
```bash
python hccro_integrated_system.py
```
This runs the full 4-layer cognitive hierarchy against 4 SOTA baselines and generates `hccro_benchmark_results.csv` and `cke_database.db`.

### 3. Running Real-World Validation Driver
Run the real-world dataset validation driver:
```bash
python scripts/run_realworld_validation.py
```

### 4. Executing Automated Test Suite
Run all unit and integration tests:
```bash
python -m unittest discover tests
```

---

## Directory Architecture

```
HCCRO/
├── config/                 # Environment & threshold configuration settings
├── data/
│   ├── raw/                # Real-World Mendeley GNSS Dataset JSON files
│   │   ├── satelliteInfomation21.json
│   │   └── pvtSolution21.json
│   └── processed/          # Processed telemetry outputs & incident logs
├── docs/                   # Final compiled design specs and developer handbooks
├── archive/                # Archived draft scripts and previous iterations
├── results/                # Output benchmark reports and evaluation CSVs
├── scripts/                # Utility scripts & real-world validation driver
├── src/                    # Primary source code package
│   ├── core/               # Data contracts, interfaces & pipeline orchestrator
│   ├── data/               # GNSS dataset parser & translation math
│   ├── hierarchy/          # 4-Layer Agent Hierarchy (Satellite, Cluster, Constellation, GS)
│   ├── optimization/       # SciPy SLSQP solver & Pareto frontier generator
│   ├── simulation/         # Telemetry streaming & orbit generators
│   ├── stages/             # 8-Stage Cognitive Resilience Pipeline implementation
│   ├── utils/              # Metrics tracker & logging utilities
│   └── validation/         # Ablation framework, SOTA baselines & experiment runner
├── tests/                  # Unit and integration test suite (19/19 passing)
├── hccro_integrated_system.py  # Production master integrated executable pipeline
├── hccro_benchmark_results.csv # Exported benchmark evaluation metrics
├── cke_database.db         # Persistent SQLite CKE knowledge database
├── README.md               # Project summary & repository introduction
└── requirements.txt        # Python package dependencies
```
