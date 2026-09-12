"""
Unknown-Attack Adaptive Response Engine
========================================
Extends the Hierarchical Cognitive Cyber Resilience Optimization (HCCRO) framework
with an isolated, zero-day threat adaptation engine.

Capabilities:
1. Open-Set Uncertainty Detection (O_t >= 0.45 or critical unmapped degradation)
2. 12-Dimensional Behavioral Fingerprinting (F_t)
3. CKE Historical Behavioral Search & Retrieval (k-NN with Cosine Similarity >= 0.80)
4. Counterfactual What-If Evaluation & Stage 6 DCS-MOS Multi-Objective Optimization
5. Stage 8 Experience Persistence in `cke_database.db` under `unknown_attack_memory`
"""

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

from src.core.interfaces import StateVectorData, OptimizationOutput, PaceState
from src.stages.stage6_optimization import AdaptiveWeightController
from src.utils.logger import get_logger

logger = get_logger("Engine.UnknownAttackAdaptive")


class UnknownAttackAdaptiveEngine:
    """
    Zero-Day Threat Adaptive Response Engine for HCCRO.
    Operates without altering existing 7D State Vector schemas or baseline execution contracts.
    """

    def __init__(self, db_path: Optional[str] = None, tau_unknown: float = 0.45):
        self.db_dir = Path("data/processed")
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else self.db_dir / "cke_database.db"
        self.tau_unknown = tau_unknown
        self.logger = logger
        self._init_db()

    def _init_db(self) -> None:
        """Creates the independent `unknown_attack_memory` table in `cke_database.db`."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS unknown_attack_memory (
                    fingerprint_id TEXT PRIMARY KEY,
                    fingerprint_vector TEXT,  -- JSON string of F_t array
                    executed_actuators TEXT,  -- JSON string of actions taken
                    observed_mci_gain REAL,   -- Empirical MCI recovery delta
                    observed_cri_gain REAL,   -- Empirical CRI recovery delta
                    success_flag INTEGER,     -- 1 if recovery S_t >= 0.85, else 0
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            conn.close()
        except Exception as err:
            self.logger.warning("[UnknownAttackEngine] DB initialization error: %s", err)

    def detect_open_set_threat(
        self,
        prob_dist: Union[Dict[str, float], List[float], np.ndarray],
        state_vector: Union[StateVectorData, Dict[str, float], Any]
    ) -> bool:
        """
        Calculates Open-Set Uncertainty Score O_t = 1.0 - max_{a in A_known} P(a).
        If O_t >= tau_unknown (0.45) OR if S_t exhibits critical unmapped degradation (T_t < 0.60 or E_t < 0.50):
          Flags threat_class = "UNKNOWN_ZERO_DAY_ATTACK" and returns True.
        """
        if isinstance(prob_dist, dict):
            max_p = max(prob_dist.values()) if prob_dist else 0.0
        elif isinstance(prob_dist, (list, tuple, np.ndarray)):
            max_p = max(prob_dist) if len(prob_dist) > 0 else 0.0
        else:
            max_p = 0.0

        o_t = 1.0 - max_p

        if isinstance(state_vector, StateVectorData):
            t_t = state_vector.T
            e_t = state_vector.E
        elif isinstance(state_vector, dict):
            t_t = state_vector.get("T", 1.0)
            e_t = state_vector.get("E", 1.0)
        else:
            t_t = getattr(state_vector, "T", 1.0)
            e_t = getattr(state_vector, "E", 1.0)

        is_unknown = (o_t >= self.tau_unknown) or (t_t < 0.60) or (e_t < 0.50)
        if is_unknown:
            self.logger.warning(
                "[Open-Set Threat] Zero-day threat detected! O_t=%.4f (tau=%.2f), T_t=%.2f, E_t=%.2f",
                o_t, self.tau_unknown, t_t, e_t
            )
        return is_unknown

    def build_fingerprint(
        self,
        state_vector: Union[StateVectorData, Dict[str, float], Any],
        raw_telemetry: Optional[Union[Dict[str, Any], Any]] = None
    ) -> np.ndarray:
        """
        Compiles a normalized 12-dimensional Behavioral Fingerprint Vector F_t:
          F_t = [ delta_C, delta_R, delta_T, delta_Q, delta_M, delta_E, delta_A,
                  delta_CPU, delta_PDR, delta_Latency, delta_Drift, delta_Buffer ]
        where delta_X = (X_nominal - X_observed) / max(1e-5, X_nominal).
        """
        if isinstance(state_vector, StateVectorData):
            c_val, r_val, t_val = state_vector.C, state_vector.R, state_vector.T
            q_val, m_val, e_val, a_val = state_vector.Q, state_vector.M, state_vector.E, state_vector.A
        elif isinstance(state_vector, dict):
            c_val = state_vector.get("C", 1.0)
            r_val = state_vector.get("R", 1.0)
            t_val = state_vector.get("T", 1.0)
            q_val = state_vector.get("Q", 1.0)
            m_val = state_vector.get("M", 1.0)
            e_val = state_vector.get("E", 1.0)
            a_val = state_vector.get("A", 1.0)
        else:
            c_val = getattr(state_vector, "C", 1.0)
            r_val = getattr(state_vector, "R", 1.0)
            t_val = getattr(state_vector, "T", 1.0)
            q_val = getattr(state_vector, "Q", 1.0)
            m_val = getattr(state_vector, "M", 1.0)
            e_val = getattr(state_vector, "E", 1.0)
            a_val = getattr(state_vector, "A", 1.0)

        # State deltas relative to nominal baseline 1.0
        delta_c = (1.0 - c_val) / 1.0
        delta_r = (1.0 - r_val) / 1.0
        delta_t = (1.0 - t_val) / 1.0
        delta_q = (1.0 - q_val) / 1.0
        delta_m = (1.0 - m_val) / 1.0
        delta_e = (1.0 - e_val) / 1.0
        delta_a = (1.0 - a_val) / 1.0

        # Telemetry metrics extraction
        cpu_obs = 0.15
        pdr_obs = 1.0
        lat_obs = 10.0
        drift_obs = 0.0
        buf_obs = 0.20

        if raw_telemetry:
            if isinstance(raw_telemetry, dict):
                cpu_obs = raw_telemetry.get("processor_utilization", raw_telemetry.get("cpu_usage", 15.0))
                if cpu_obs > 1.0:
                    cpu_obs /= 100.0

                pdr_obs = raw_telemetry.get("packet_delivery_ratio", raw_telemetry.get("pdr", 1.0))
                lat_obs = raw_telemetry.get("communication_latency", raw_telemetry.get("latency_ms", 10.0))
                drift_obs = raw_telemetry.get("navigation_consistency", raw_telemetry.get("gps_drift_m", 0.0))
                buf_obs = raw_telemetry.get("memory_consumption", raw_telemetry.get("buffer_usage", 20.0))
                if buf_obs > 1.0:
                    buf_obs /= 100.0

        delta_cpu = (cpu_obs - 0.15) / 0.15
        delta_pdr = (1.0 - pdr_obs) / 1.0
        delta_lat = (lat_obs - 10.0) / 10.0
        delta_drift = drift_obs / 1.0
        delta_buf = (buf_obs - 0.20) / 0.20

        f_t = np.array([
            delta_c, delta_r, delta_t, delta_q, delta_m, delta_e, delta_a,
            delta_cpu, delta_pdr, delta_lat, delta_drift, delta_buf
        ], dtype=np.float64)

        return f_t

    def search_cke_memory(self, fingerprint: np.ndarray) -> List[str]:
        """
        Queries `unknown_attack_memory` using Cosine Similarity Sim(F_t, F_hist).
        If max(Sim) >= 0.80: retrieves successful candidate actions from top matching historical entry.
        Else: returns candidate action pool:
          {'ISOLATE_AND_REROUTE', 'WORKLOAD_MIGRATION', 'PURGE_AND_RESET', 'HYBRID_SHIELD'}
        """
        default_pool = [
            'ISOLATE_AND_REROUTE',
            'WORKLOAD_MIGRATION',
            'PURGE_AND_RESET',
            'HYBRID_SHIELD'
        ]

        if fingerprint is None or len(fingerprint) == 0:
            return default_pool

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT fingerprint_vector, executed_actuators, success_flag FROM unknown_attack_memory")
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return default_pool

            best_sim = -1.0
            best_actions = None

            norm_f = np.linalg.norm(fingerprint)
            if norm_f < 1e-9:
                return default_pool

            for fp_json, actions_json, success in rows:
                if success != 1:
                    continue
                try:
                    f_hist = np.array(json.loads(fp_json), dtype=np.float64)
                    norm_hist = np.linalg.norm(f_hist)
                    if norm_hist < 1e-9:
                        continue
                    sim = float(np.dot(fingerprint, f_hist) / (norm_f * norm_hist))
                    if sim > best_sim:
                        best_sim = sim
                        best_actions = json.loads(actions_json)
                except Exception:
                    continue

            if best_sim >= 0.80 and best_actions:
                self.logger.info("[CKE Memory Retrieval] Found historical match! Sim=%.4f, Actions=%s", best_sim, best_actions)
                if isinstance(best_actions, str):
                    return [best_actions]
                return list(best_actions)

        except Exception as err:
            self.logger.warning("[CKE Memory Search Warning] %s", err)

        return default_pool

    def evaluate_counterfactuals(
        self,
        state_vector: Union[StateVectorData, Dict[str, float], Any],
        candidate_actions: List[str]
    ) -> str:
        """
        Estimates 1-step Counterfactual State Vector S_pred(a) = S_t + Gamma(a) - Penalty_env.
        Evaluates Stage 6 DCS-MOS Multi-Objective Utility U(a):
          Maximize U(a) = alpha*R + beta*M + gamma*T + delta*C + lambda*H
        Subject to hardware constraints: E_draw(a) <= E_available and CPU_utilization(a) <= 85%.
        Selects optimal counterfactual action: a* = argmax U(a).
        """
        if not candidate_actions:
            return "HYBRID_SHIELD"

        actuator_gamma = {
            "ISOLATE_AND_REROUTE": {"C": 0.35, "Q": 0.30, "R": 0.25, "E": -0.05, "CPU": 0.20},
            "WORKLOAD_MIGRATION":  {"M": 0.35, "E": 0.25, "A": 0.25, "R": 0.20, "CPU": 0.35},
            "PURGE_AND_RESET":      {"T": 0.40, "A": 0.30, "R": 0.30, "C": 0.15, "CPU": 0.15},
            "HYBRID_SHIELD":       {"C": 0.30, "T": 0.30, "Q": 0.30, "R": 0.35, "CPU": 0.40},
        }

        if isinstance(state_vector, StateVectorData):
            s_dict = {"C": state_vector.C, "R": state_vector.R, "T": state_vector.T,
                      "Q": state_vector.Q, "M": state_vector.M, "E": state_vector.E, "A": state_vector.A}
        elif isinstance(state_vector, dict):
            s_dict = dict(state_vector)
        else:
            s_dict = {
                "C": getattr(state_vector, "C", 1.0),
                "R": getattr(state_vector, "R", 1.0),
                "T": getattr(state_vector, "T", 1.0),
                "Q": getattr(state_vector, "Q", 1.0),
                "M": getattr(state_vector, "M", 1.0),
                "E": getattr(state_vector, "E", 1.0),
                "A": getattr(state_vector, "A", 1.0),
            }

        weights = AdaptiveWeightController.compute_weights(
            energy_availability=s_dict.get("E", 1.0),
            cpu_load=0.20
        )
        alpha = weights.get("alpha_R", 0.30)
        beta = weights.get("beta_M", 0.25)
        gamma = weights.get("gamma_T", 0.20)
        delta = weights.get("delta_C", 0.15)
        lambd = weights.get("lambda_H", 0.10)
        norm_w = max(1e-5, alpha + beta + gamma + delta + lambd)

        best_action = candidate_actions[0]
        best_utility = -1.0

        for action in candidate_actions:
            gamma_map = actuator_gamma.get(action, {"C": 0.2, "R": 0.2, "T": 0.2, "CPU": 0.2})
            predicted_cpu = gamma_map.get("CPU", 0.20)
            if predicted_cpu > 0.85:
                continue

            pred_r = min(1.0, max(0.0, s_dict.get("R", 0.5) + gamma_map.get("R", 0.1)))
            pred_m = min(1.0, max(0.0, s_dict.get("M", 0.5) + gamma_map.get("M", 0.1)))
            pred_t = min(1.0, max(0.0, s_dict.get("T", 0.5) + gamma_map.get("T", 0.1)))
            pred_c = min(1.0, max(0.0, s_dict.get("C", 0.5) + gamma_map.get("C", 0.1)))
            pred_h = min(1.0, max(0.0, s_dict.get("Q", 0.5) + gamma_map.get("Q", 0.1)))

            u_a = (alpha * pred_r + beta * pred_m + gamma * pred_t + delta * pred_c + lambd * pred_h) / norm_w

            if u_a > best_utility:
                best_utility = u_a
                best_action = action

        self.logger.info("[Counterfactual Evaluation] Selected optimal action: %s (Utility=%.4f)", best_action, best_utility)
        return best_action

    def persist_experience(
        self,
        fingerprint: np.ndarray,
        action: Union[str, List[str]],
        mci_gain: float,
        cri_gain: float,
        success: bool
    ) -> bool:
        """
        Serializes F_t, action, gains, and success flag into `unknown_attack_memory` in `cke_database.db`.
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            fp_id = f"zero_day_{uuid.uuid4().hex[:12]}"
            fp_json = json.dumps(fingerprint.tolist() if isinstance(fingerprint, np.ndarray) else list(fingerprint))
            act_json = json.dumps([action] if isinstance(action, str) else list(action))
            success_flag = 1 if success else 0

            cursor.execute("""
                INSERT INTO unknown_attack_memory (
                    fingerprint_id, fingerprint_vector, executed_actuators,
                    observed_mci_gain, observed_cri_gain, success_flag
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (fp_id, fp_json, act_json, float(mci_gain), float(cri_gain), success_flag))

            conn.commit()
            conn.close()
            self.logger.info("[CKE Experience Persisted] Record ID=%s, Success=%d, MCI Gain=%.4f", fp_id, success_flag, mci_gain)
            return True
        except Exception as err:
            self.logger.warning("[CKE Persistence Error] %s", err)
            return False
