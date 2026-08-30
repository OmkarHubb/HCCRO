"""
Stage 8: Cyber Knowledge Evolution (CKE)
=========================================
Logs incident profiles, telemetry indicators, threat intents, applied weights, and healing outcomes
to a persistent local database (JSON / SQLite) and provides feedback lookup queries for future cycles.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, Union, Optional
from src.core.interfaces import BaseStage, StateVectorData, OptimizationOutput, HealingOutput, CKEOutput
from src.utils.logger import get_logger

logger = get_logger("Stage8.CKE")


class CyberKnowledgeEvolution(BaseStage):
    """
    Stage 8: Cyber Knowledge Evolution (CKE) Engine.
    Inherits from BaseStage to enforce plug-and-play adaptability.
    """

    def __init__(self, db_path: Optional[str] = None):
        super().__init__(name="Stage8_CKE")
        self.db_dir = Path("data/processed")
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.json_file = self.db_dir / "cke_history.json"
        self.sqlite_file = self.db_dir / "cke_history.db"
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initializes local SQLite database table for persistent audit logs."""
        try:
            conn = sqlite3.connect(str(self.sqlite_file))
            cursor = conn.cursor()
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
            logger.warning("[Stage 8 CKE] SQLite initialization fallback: %s", err)

    def get_historical_success_rate(
        self, attack_intent: str, mitigation_action: str
    ) -> float:
        """
        Feedback query interface used by Stages 3 and 6 to refine future prediction and optimization cycles.
        
        Args:
            attack_intent: Threat intention category (e.g. 'RF_JAMMING_DISRUPTION').
            mitigation_action: Countermeasure action key (e.g. 'ACTIVATE_FREQUENCY_HOPPING').
            
        Returns:
            float: Historical mitigation success rate [0.0, 1.0].
        """
        try:
            conn = sqlite3.connect(str(self.sqlite_file))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM cke_incidents WHERE active_threats LIKE ? AND optimal_actions LIKE ? AND status = 'SUCCESS'",
                (f"%{attack_intent}%", f"%{mitigation_action}%"),
            )
            success_count = cursor.fetchone()[0]
            cursor.execute(
                "SELECT COUNT(*) FROM cke_incidents WHERE active_threats LIKE ? AND optimal_actions LIKE ?",
                (f"%{attack_intent}%", f"%{mitigation_action}%"),
            )
            total_count = cursor.fetchone()[0]
            conn.close()

            if total_count > 0:
                return round(float(success_count) / float(total_count), 3)
        except Exception:
            pass

        return 0.95  # Nominal high fallback success rate

    def execute(self, input_data: Dict[str, Any]) -> CKEOutput:
        """
        Executes Stage 8 persistent logging and knowledge evolution.
        
        Args:
            input_data: Dict containing 'state_vector', 'optimization_output', and 'healing_output'.
            
        Returns:
            CKEOutput: Standardized knowledge record artifact.
        """
        s_t = input_data.get("state_vector") or StateVectorData()
        opt = input_data.get("optimization_output") or OptimizationOutput([], s_t.pace_state, 1.0, 1.0, {})
        heal = input_data.get("healing_output") or HealingOutput("SUCCESS", s_t.pace_state, [], 0.0, {})

        record_id = f"CKE_LOG_{int(s_t.timestamp)}"
        threats_str = ",".join(s_t.active_threats) if s_t.active_threats else "NOMINAL"
        actions_str = ",".join(opt.optimal_actions) if opt.optimal_actions else "NONE"
        pace_str = heal.current_pace_state.value if hasattr(heal.current_pace_state, "value") else str(heal.current_pace_state)

        persisted = False

        # SQLite Persistence
        try:
            conn = sqlite3.connect(str(self.sqlite_file))
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO cke_incidents
                (record_id, timestamp, active_threats, pace_state, optimal_actions, status, utility_score, drei_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record_id,
                    s_t.timestamp,
                    threats_str,
                    pace_str,
                    actions_str,
                    heal.status,
                    opt.utility_score,
                    opt.drei_score,
                ),
            )
            conn.commit()
            conn.close()
            persisted = True
        except Exception as err:
            logger.warning("[Stage 8 CKE] SQLite log insert warning: %s", err)

        # Flat JSON Persistence
        try:
            records = []
            if self.json_file.exists() and self.json_file.stat().st_size > 0:
                with open(self.json_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            records.append({
                "record_id": record_id,
                "timestamp": s_t.timestamp,
                "threats": s_t.active_threats,
                "pace_state": pace_str,
                "actions": opt.optimal_actions,
                "status": heal.status,
                "utility_score": opt.utility_score,
                "drei_score": opt.drei_score,
            })
            with open(self.json_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            persisted = True
        except Exception as err:
            logger.warning("[Stage 8 CKE] JSON log write warning: %s", err)

        hist_rate = self.get_historical_success_rate(threats_str, actions_str)

        logger.info("[Stage 8 CKE] Persisted audit record '%s' (Status: %s).", record_id, heal.status)

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
