"""
Real-World Mendeley GNSS Dataset Parser & Mathematical State Translator
========================================================================
Parses parallel columnar JSON telemetry files from active cyber-attack campaign
(December 21, 2023 - Hour 21):
- satelliteInfomation21.json
- pvtSolution21.json

Translates raw physical observations (C/N0, ECEF coordinates, RAIM satellite usage, gDOP)
into the 7-Dimensional State Vector S_t = {C, R, T, Q, M, E, A} for Stage 1 CSA.
"""

import json
import math
import os
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from src.core.interfaces import StateVectorData, TelemetryData, PaceState
from src.utils.logger import get_logger

logger = get_logger("Data.GNSSParser")


class RealWorldGNSSParser:
    """
    Parser and translator for the Mendeley GNSS Real-World Dataset (Hour 21 active campaign).
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.sat_file = self._find_file("satelliteInfomation21.json")
        self.pvt_file = self._find_file("pvtSolution21.json")
        
        self.parsed_records: List[Dict[str, Any]] = []
        self._df: Optional[pd.DataFrame] = None
        self._is_loaded = False

    def _find_file(self, filename: str) -> str:
        """Locates source JSON file across candidate directory paths."""
        candidates = [
            os.path.join(self.data_dir, "raw", filename),
            os.path.join(self.data_dir, filename),
            os.path.join(os.getcwd(), "data", "raw", filename),
            os.path.join(os.getcwd(), "data", filename),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", filename)),
        ]
        for path in candidates:
            if os.path.exists(path):
                logger.info("[GNSSParser] Found dataset file: %s", path)
                return path
        raise FileNotFoundError(f"Could not locate '{filename}' in candidate paths: {candidates}")

    def load_and_parse(self) -> List[Dict[str, Any]]:
        """
        Parses parallel columnar JSON files and translates raw metrics to S_t state vectors.
        
        Returns:
            List of parsed dictionary records per temporal epoch t.
        """
        logger.info("[GNSSParser] Loading JSON files: %s & %s", self.sat_file, self.pvt_file)
        with open(self.sat_file, "r", encoding="utf-8") as f_sat:
            sat_json = json.load(f_sat)
        with open(self.pvt_file, "r", encoding="utf-8") as f_pvt:
            pvt_json = json.load(f_pvt)

        # Align timeline by recordTime / timestamp
        timestamps = sat_json.get("recordTime") or sat_json.get("timestamp") or []
        num_epochs = len(timestamps)
        logger.info("[GNSSParser] Ingesting %d parallel temporal epochs...", num_epochs)

        # Baseline ECEF coordinates at t=0 (Yunnan University Science Hall)
        # Note: Raw ECEF in pvtSolution21 is in centimeters (divide by 100.0 to get meters)
        x0_raw = pvt_json["ecefX"][0]
        y0_raw = pvt_json["ecefY"][0]
        z0_raw = pvt_json["ecefZ"][0]
        
        scale_factor = 100.0 if abs(x0_raw) > 1e7 else 1.0
        x0 = x0_raw / scale_factor
        y0 = y0_raw / scale_factor
        z0 = z0_raw / scale_factor

        self.parsed_records = []

        for t in range(num_epochs):
            ts = timestamps[t]
            
            # --- 1. Communication Quality Q_t Mapping (RF Jamming Detector) ---
            cno_G = sat_json["cno_G"][t]
            active_cnos = [c for c in cno_G if c > 0.0]
            
            if not active_cnos:
                avg_cno_t = 0.0
                q_t = 0.0
            else:
                avg_cno_t = sum(active_cnos) / len(active_cnos)
                # Sigmoidal scaling centered at 25 dB-Hz
                q_t = 1.0 / (1.0 + math.exp(-(avg_cno_t - 25.0) / 3.0))

            q_t = round(min(1.0, max(0.0, q_t)), 4)

            # --- 2. Trust State T_t Mapping (GPS Spoofing & RAIM Rejection Detector) ---
            # a) Spatial ECEF Drift
            x_t = pvt_json["ecefX"][t] / scale_factor
            y_t = pvt_json["ecefY"][t] / scale_factor
            z_t = pvt_json["ecefZ"][t] / scale_factor
            drift_meters_t = math.sqrt((x_t - x0)**2 + (y_t - y0)**2 + (z_t - z0)**2)
            drift_meters_t = round(drift_meters_t, 4)

            # b) SV Tracking Ratio (RAIM Anomaly Detector)
            sv_used = sat_json["svUsed_G"][t]
            sv_ids = sat_json["svId_G"][t]
            used_cnt = sum(1 for u in sv_used if u >= 0.9)
            visible_cnt = max(1, sum(1 for sv in sv_ids if sv > 0.0))
            sv_ratio_t = min(1.0, max(0.0, used_cnt / visible_cnt))
            sv_ratio_t = round(sv_ratio_t, 4)

            # c) Dilution of Precision Penalty
            gdop_t = float(pvt_json["gDOP"][t])
            if gdop_t > 1.5:
                dop_penalty_t = 1.5 / gdop_t
            else:
                dop_penalty_t = 1.0
            dop_penalty_t = round(dop_penalty_t, 4)

            # d) Consolidated Trust State
            t_t = math.exp(-drift_meters_t / 100.0) * sv_ratio_t * dop_penalty_t
            t_t = round(min(1.0, max(0.0, t_t)), 4)

            # --- 3. Active Threat Identification & State Vector Construction ---
            active_threats = []
            if q_t < 0.5:
                active_threats.append("RF_JAMMING_SUSPECTED")
            if t_t < 0.5 or drift_meters_t > 50.0:
                active_threats.append("GPS_SPOOFING_SUSPECTED")

            # Consolidated 7-D State Vector S_t = {C, R, T, Q, M, E, A}
            c_val = q_t
            t_val = t_t
            q_val = q_t
            m_val = sv_ratio_t
            e_val = round(max(0.1, min(1.0, 1.0 - (1.0 - sv_ratio_t) * 0.3)), 4)
            a_val = 1.0
            r_val = round((c_val + t_val + q_val + e_val) / 4.0, 4)

            # Assign PACE State
            if r_val < 0.30 or len(active_threats) >= 2:
                pace = PaceState.EMERGENCY
            elif r_val < 0.55 or len(active_threats) == 1:
                pace = PaceState.CONTINGENCY
            elif r_val < 0.75:
                pace = PaceState.ALTERNATE
            else:
                pace = PaceState.PRIMARY

            s_t = StateVectorData(
                C=c_val,
                R=r_val,
                T=t_val,
                Q=q_val,
                M=m_val,
                E=e_val,
                A=a_val,
                pace_state=pace,
                active_threats=active_threats,
            )

            record = {
                "epoch_index": t,
                "timestamp": ts,
                "lat": pvt_json["lat"][t],
                "lon": pvt_json["lon"][t],
                "height": pvt_json["height"][t],
                "ecefX": x_t,
                "ecefY": y_t,
                "ecefZ": z_t,
                "gDOP": gdop_t,
                "numSvs": sat_json["numSvs"][t],
                "active_cnos_count": len(active_cnos),
                "avg_cno": round(avg_cno_t, 2),
                "drift_meters": drift_meters_t,
                "sv_ratio": sv_ratio_t,
                "dop_penalty": dop_penalty_t,
                "Q_t": q_t,
                "T_t": t_t,
                "S_t": s_t,
                "active_threats": active_threats,
                "pace_state": pace.value,
            }
            self.parsed_records.append(record)

        self._is_loaded = True
        logger.info("[GNSSParser] Successfully parsed and translated %d epochs.", len(self.parsed_records))
        return self.parsed_records

    def get_epoch(self, t: int) -> Dict[str, Any]:
        """Returns parsed telemetry record for epoch t."""
        if not self._is_loaded:
            self.load_and_parse()
        if 0 <= t < len(self.parsed_records):
            return self.parsed_records[t]
        raise IndexError(f"Epoch index {t} out of range [0, {len(self.parsed_records)-1}].")

    def get_state_vector(self, t: int) -> StateVectorData:
        """Returns strongly-typed StateVectorData S_t for epoch t."""
        return self.get_epoch(t)["S_t"]

    def to_dataframe(self) -> pd.DataFrame:
        """Returns full dataset as a clean pandas DataFrame."""
        if not self._is_loaded:
            self.load_and_parse()
        if self._df is None:
            df_records = []
            for r in self.parsed_records:
                d = dict(r)
                d.pop("S_t", None)
                df_records.append(d)
            self._df = pd.DataFrame(df_records)
        return self._df

    def get_summary(self) -> Dict[str, Any]:
        """Computes executive summary statistics across the parsed dataset."""
        if not self._is_loaded:
            self.load_and_parse()
        
        drifts = [r["drift_meters"] for r in self.parsed_records]
        q_vals = [r["Q_t"] for r in self.parsed_records]
        t_vals = [r["T_t"] for r in self.parsed_records]
        threat_count = sum(1 for r in self.parsed_records if r["active_threats"])

        return {
            "total_epochs": len(self.parsed_records),
            "peak_drift_meters": max(drifts),
            "min_q_t": min(q_vals),
            "mean_q_t": round(sum(q_vals) / len(q_vals), 4),
            "min_t_t": min(t_vals),
            "mean_t_t": round(sum(t_vals) / len(t_vals), 4),
            "active_threat_epochs": threat_count,
        }
