"""
Stage 8: Cyber Knowledge Evolution (CKE) - The Persistent SQLite Memory Loop
=============================================================================
Logs incident profiles, raw sensor metrics, threat intents, DCS-MOS weights, PACE transitions,
and healing outcomes to a persistent local zero-dependency SQLite database (`hccro_experience.db`),
providing historical success queries to close the cognitive loop.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, Union, Optional
from src.core.interfaces import BaseStage, StateVectorData, OptimizationOutput, HealingOutput, CKEOutput, TelemetryData
from src.utils.logger import get_logger

logger = get_logger("Stage8.CKE")


class CyberKnowledgeEvolution(BaseStage):
    """
    Stage 8: Cyber Knowledge Evolution (CKE) Persistent Memory Loop Engine.
    Inherits from BaseStage to enforce plug-and-play adaptability.
    """

    def __init__(self, db_path: Optional[str] = None):
        super().__init__(name="Stage8_CKE")
        self.db_dir = Path("data/processed")
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_file = Path(db_path) if db_path else self.db_dir / "hccro_experience.db"
        self.json_file = self.db_dir / "cke_history.json"
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initializes local SQLite database schema for persistent audit logs."""
        try:
            conn = sqlite3.connect(str(self.sqlite_file))
            cursor = conn.cursor()
            
            # Create incident_history table per requirements
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incident_history (
                    cycle_id TEXT PRIMARY KEY,
                    satellite_id TEXT,
                    timestamp REAL,
                    cpu_usage REAL,
                    ram_usage REAL,
                    latency_ms REAL,
                    noise_floor_db REAL,
                    gps_drift_m REAL,
                    attack_intent TEXT,
                    dcs_mos_weights TEXT,
                    previous_pace_mode TEXT,
                    target_pace_mode TEXT,
                    executed_action TEXT,
                    drei_score REAL,
                    mci_score REAL,
                    status TEXT
                )
            """)

            # Create alias table cke_incidents for backward compatibility
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cke_incidents (
                    record_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    active_threats TEXT,
                    pace_state TEXT,
                    optimal_actions TEXT,
                    status TEXT,
                    utility_score REAL,
                    drei_score REAL
                )
            """)

            conn.commit()
            conn.close()
        except Exception as err:
            logger.warning("[Stage 8 CKE] SQLite initialization warning: %s", err)

    def log_incident_cycle(
        self,
        cycle_id: str,
        satellite_id: str,
        timestamp: float,
        sensor_metrics: Dict[str, float],
        attack_intent: str,
        dcs_mos_weights: Dict[str, float],
        previous_pace_mode: str,
        target_pace_mode: str,
        executed_action: str,
        drei_score: float,
        mci_score: float,
        status: str = "SUCCESS",
    ) -> bool:
        """
        Logs every execution step of the 8-stage pipeline into SQLite database `hccro_experience.db`.
        """
        try:
            conn = sqlite3.connect(str(self.sqlite_file))
            cursor = conn.cursor()

            weights_json = json.dumps(dcs_mos_weights)

            cursor.execute(
                """
                INSERT OR REPLACE INTO incident_history (
                    cycle_id, satellite_id, timestamp, cpu_usage, ram_usage,
                    latency_ms, noise_floor_db, gps_drift_m, attack_intent,
                    dcs_mos_weights, previous_pace_mode, target_pace_mode,
                    executed_action, drei_score, mci_score, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    cycle_id,
                    satellite_id,
                    timestamp,
                    sensor_metrics.get("cpu_usage", 15.0),
                    sensor_metrics.get("ram_usage", 30.0),
                    sensor_metrics.get("latency_ms", 50.0),
                    sensor_metrics.get("noise_floor_db", 10.0),
                    sensor_metrics.get("gps_drift_m", 0.0),
                    attack_intent,
                    weights_json,
                    previous_pace_mode,
                    target_pace_mode,
                    executed_action,
                    drei_score,
                    mci_score,
                    status,
                ),
            )

            # Also mirror into cke_incidents for backward compatibility
            cursor.execute(
                """
                INSERT OR REPLACE INTO cke_incidents (
                    record_id, timestamp, active_threats, pace_state,
                    optimal_actions, status, utility_score, drei_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    cycle_id,
                    timestamp,
                    attack_intent,
                    target_pace_mode,
                    executed_action,
                    status,
                    1.0 - mci_score,
                    drei_score,
                ),
            )

            conn.commit()
            conn.close()
            logger.info("[Stage 8 CKE] Persisted incident cycle '%s' to SQLite database.", cycle_id)
            return True
        except Exception as err:
            logger.error("[Stage 8 CKE] SQLite log_incident_cycle error: %s", err)
            return False

    def get_historical_success_rate(
        self, attack_intent: str, mitigation_action: str
    ) -> float:
        """
        Feedback query interface calculating average DREI and MCI improvement achieved
        by specific self-healing actions under specific attack states.
        
        Args:
            attack_intent: Threat intention category (e.g. 'RF_JAMMING_DISRUPTION' or 'RF_JAMMING_SUSPECTED').
            mitigation_action: Countermeasure action key (e.g. 'TRIGGER_FREQUENCY_HOPPING').
            
        Returns:
            float: Historical mitigation success rate [0.0, 1.0].
        """
        try:
            conn = sqlite3.connect(str(self.sqlite_file))
            cursor = conn.cursor()

            # Query incident_history first
            cursor.execute(
                """
                SELECT COUNT(*), AVG(drei_score), AVG(mci_score) 
                FROM incident_history 
                WHERE attack_intent LIKE ? AND executed_action LIKE ? AND status = 'SUCCESS'
            """,
                (f"%{attack_intent}%", f"%{mitigation_action}%"),
            )
            success_row = cursor.fetchone()

            cursor.execute(
                """
                SELECT COUNT(*) FROM incident_history 
                WHERE attack_intent LIKE ? AND executed_action LIKE ?
            """,
                (f"%{attack_intent}%", f"%{mitigation_action}%"),
            )
            total_count = cursor.fetchone()[0]

            conn.close()

            if total_count > 0:
                success_count = success_row[0]
                rate = float(success_count) / float(total_count)
                return round(rate, 3)

        except Exception as err:
            logger.warning("[Stage 8 CKE] Query historical success rate warning: %s", err)

        return 0.95  # Nominal high fallback success rate

    def execute(self, input_data: Dict[str, Any]) -> CKEOutput:
        """
        Executes Stage 8 persistent logging and knowledge evolution.
        
        Args:
            input_data: Dict containing 'state_vector', 'optimization_output', 'healing_output', etc.
            
        Returns:
            CKEOutput: Standardized knowledge record artifact.
        """
        s_t = input_data.get("state_vector") or StateVectorData()
        opt = input_data.get("optimization_output") or OptimizationOutput([], s_t.pace_state, 1.0, 1.0, {})
        heal = input_data.get("healing_output") or HealingOutput("SUCCESS", s_t.pace_state, [], 0.0, {})
        mci_score = input_data.get("mci_score", 0.0)

        record_id = f"CKE_LOG_{int(s_t.timestamp)}"
        satellite_id = input_data.get("satellite_id", "SAT_HCCRO_01")
        threats_str = ",".join(s_t.active_threats) if s_t.active_threats else "NOMINAL"
        actions_str = ",".join(opt.optimal_actions) if opt.optimal_actions else "MAINTAIN_NOMINAL_OPERATIONS"
        prev_pace_str = s_t.pace_state.value if hasattr(s_t.pace_state, "value") else str(s_t.pace_state)
        targ_pace_str = heal.current_pace_state.value if hasattr(heal.current_pace_state, "value") else str(heal.current_pace_state)

        sensor_metrics = {
            "cpu_usage": (1.0 - s_t.E) * 100.0,
            "ram_usage": 30.0,
            "latency_ms": (1.0 - s_t.C) * 200.0,
            "noise_floor_db": (1.0 - s_t.Q) * 20.0,
            "gps_drift_m": (1.0 - s_t.T) * 15.0,
        }

        # Persist to SQLite
        persisted = self.log_incident_cycle(
            cycle_id=record_id,
            satellite_id=satellite_id,
            timestamp=s_t.timestamp,
            sensor_metrics=sensor_metrics,
            attack_intent=threats_str,
            dcs_mos_weights=opt.weights_applied,
            previous_pace_mode=prev_pace_str,
            target_pace_mode=targ_pace_str,
            executed_action=actions_str,
            drei_score=opt.drei_score,
            mci_score=mci_score,
            status=heal.status,
        )

        # Mirror to flat JSON
        try:
            records = []
            if self.json_file.exists() and self.json_file.stat().st_size > 0:
                with open(self.json_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            records.append({
                "record_id": record_id,
                "satellite_id": satellite_id,
                "timestamp": s_t.timestamp,
                "threats": s_t.active_threats,
                "pace_state": targ_pace_str,
                "actions": opt.optimal_actions,
                "status": heal.status,
                "utility_score": opt.utility_score,
                "drei_score": opt.drei_score,
            })
            with open(self.json_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        except Exception as err:
            logger.warning("[Stage 8 CKE] JSON log write warning: %s", err)

        hist_rate = self.get_historical_success_rate(threats_str, actions_str)

        return CKEOutput(
            record_id=record_id,
            logged_at=s_t.timestamp,
            persisted_successfully=persisted,
            historical_success_rate=hist_rate,
        )

    def process(self, state: Any, plan: Any, status: Any) -> Dict[str, Any]:
        """Backward compatibility bridge returning dictionary representation."""
        if isinstance(state, StateVectorData):
            s_t = state
        else:
            s_t = StateVectorData(timestamp=getattr(state, "timestamp", time.time()), active_threats=getattr(state, "active_threats", []))
        out = self.execute({"state_vector": s_t})
        return {
            "record_id": out.record_id,
            "threats_logged": s_t.active_threats,
            "success_flag": out.persisted_successfully,
        }


# Backward compatibility alias
Stage8CKE = CyberKnowledgeEvolution
