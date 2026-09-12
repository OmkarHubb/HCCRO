"""
Real-World ESA OPSSAT-AD Dataset Parser & Mathematical State Translator
========================================================================
Parses segment-level statistical telemetry features from the ESA OPS-SAT
Anomaly Detection (OPSSAT-AD) Malicious Telecommand Injection & Protocol
Anomaly campaign dataset.

Translates raw CSV columns (channel, anomaly, diff_peaks, var_div_duration,
std, gaps_squared) into the 7-Dimensional State Vector
S_t = {C, R, T, Q, M, E, A} bounded in [0.0, 1.0] for Stage 1 CSA.

Dataset source: data/processed/dataset.csv
  - 2,123 segments across 9 CADC satellite subsystem channels
  - anomaly == 1 indicates active malicious telecommand injection (434 segments)
  - anomaly == 0 indicates nominal operation (1,689 segments)
"""

import os
from typing import Dict, Any, List, Optional

import pandas as pd

from src.core.interfaces import StateVectorData, PaceState
from src.utils.logger import get_logger

logger = get_logger("Data.OPSSATParser")


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Strictly clamps value to [lo, hi]."""
    return max(lo, min(hi, float(value)))


class RealWorldOPSSATParser:
    """
    Production parser and translator for the ESA OPSSAT-AD Malicious
    Telecommand Injection & Protocol Anomaly campaign dataset.

    Converts raw CSV segment-level statistical telemetry features into
    the continuous 7-Dimensional State Vector S_t = {C, R, T, Q, M, E, A}.

    API mirrors RealWorldGNSSParser for consistency across data modules.
    """

    # Required CSV columns for S_t translation
    REQUIRED_COLUMNS = [
        "segment", "anomaly", "channel",
        "diff_peaks", "var_div_duration", "std", "gaps_squared",
    ]

    def __init__(self, csv_path: str = "data/processed/dataset.csv"):
        self.csv_path = self._resolve_csv_path(csv_path)
        self.parsed_records: List[Dict[str, Any]] = []
        self._df: Optional[pd.DataFrame] = None
        self._raw_df: Optional[pd.DataFrame] = None
        self._is_loaded: bool = False

    def _resolve_csv_path(self, csv_path: str) -> str:
        """Locates the OPSSAT-AD CSV across candidate directory paths."""
        candidates = [
            csv_path,
            os.path.join(os.getcwd(), csv_path),
            os.path.join(os.getcwd(), "data", "processed", "dataset.csv"),
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "dataset.csv")
            ),
        ]
        for path in candidates:
            if os.path.exists(path):
                logger.info("[OPSSATParser] Found dataset file: %s", path)
                return path
        raise FileNotFoundError(
            f"Could not locate OPSSAT-AD dataset in candidate paths: {candidates}"
        )

    # ------------------------------------------------------------------
    # Core Mathematical S_t Translation Engine
    # ------------------------------------------------------------------

    @staticmethod
    def compute_state_vector(
        anomaly: int,
        diff_peaks: float,
        var_div_duration: float,
        std_val: float,
        gaps_squared: float,
        channel: str = "",
        segment_index: int = 0,
    ) -> StateVectorData:
        """
        Converts a single raw CSV segment into a 7-D State Vector
        S_t = {C, R, T, Q, M, E, A} bounded in [0.0, 1.0].

        Mathematical Translation Engine (per specification):
        1. A_t = clamp(1.0 - (anomaly * 0.70) - 0.30 * min(1.0, diff_peaks / 100.0))
        2. E_t = clamp(1.0 - (anomaly * 0.50) - 0.50 * min(1.0, var_div_duration / 1e-11))
        3. T_t = clamp(1.0 - (anomaly * 0.85) - 0.15 * min(1.0, std / 1e-4))
        4. C_t = clamp(1.0 - (anomaly * 0.80))
        5. Q_t = clamp(1.0 - min(1.0, gaps_squared / 1000.0))
        6. R_t = 1.0 if anomaly == 0 else 0.30
        7. M_t = clamp(E_t * A_t)

        Args:
            anomaly: Ground-truth indicator (1 = malicious injection, 0 = nominal).
            diff_peaks: Peak derivative count (command execution jitter).
            var_div_duration: Normalized variance density (buffer instability).
            std_val: Standard deviation of signal segment (telemetry variance).
            gaps_squared: Telemetry frame gaps squared (link drops).
            channel: Subsystem channel name (e.g., 'CADC0872').
            segment_index: Segment index for timestamp derivation.

        Returns:
            StateVectorData: Strongly-typed 7-D state vector.
        """
        anom = int(anomaly)

        # 1. Autonomous Decision Reliability (A_t) — Command Execution Integrity
        a_t = _clamp(
            1.0 - (anom * 0.70) - 0.30 * min(1.0, diff_peaks / 100.0)
        )

        # 2. Resource Efficiency (E_t) — Onboard Buffer & Compute Health
        e_t = _clamp(
            1.0 - (anom * 0.50) - 0.50 * min(1.0, var_div_duration / 1e-11)
        )

        # 3. Trust State (T_t) — Host & Subsystem Credibility
        t_t = _clamp(
            1.0 - (anom * 0.85) - 0.15 * min(1.0, std_val / 1e-4)
        )

        # 4. Communication Integrity (C_t) & Communication Quality (Q_t)
        c_t = _clamp(1.0 - (anom * 0.80))
        q_t = _clamp(1.0 - min(1.0, gaps_squared / 1000.0))

        # 5. Routing Stability (R_t) & Mission Performance (M_t)
        r_t = 1.0 if anom == 0 else 0.30
        m_t = _clamp(e_t * a_t)

        # Round to 4 decimal places for consistency with pipeline conventions
        c_t = round(c_t, 4)
        r_t = round(r_t, 4)
        t_t = round(t_t, 4)
        q_t = round(q_t, 4)
        m_t = round(m_t, 4)
        e_t = round(e_t, 4)
        a_t = round(a_t, 4)

        # Active threat identification
        active_threats: List[str] = []
        if anom == 1:
            active_threats.append("TELECOMMAND_INJECTION_SUSPECTED")
            # CADC channel anomalies indicate ADCS/Attitude subsystem compromise
            if channel.upper().startswith("CADC"):
                active_threats.append("CADC_ANOMALY_SUSPECTED")

        # PACE State assignment based on composite health and threat severity
        if len(active_threats) >= 2 or (a_t < 0.30 and t_t < 0.30):
            pace = PaceState.EMERGENCY
        elif anom == 1 or a_t < 0.50 or t_t < 0.50:
            pace = PaceState.CONTINGENCY
        elif a_t < 0.75 or t_t < 0.75:
            pace = PaceState.ALTERNATE
        else:
            pace = PaceState.PRIMARY

        return StateVectorData(
            C=c_t,
            R=r_t,
            T=t_t,
            Q=q_t,
            M=m_t,
            E=e_t,
            A=a_t,
            pace_state=pace,
            timestamp=float(segment_index),
            active_threats=active_threats,
        )

    # ------------------------------------------------------------------
    # Dataset Loading & Parsing
    # ------------------------------------------------------------------

    def load_and_parse(self) -> List[Dict[str, Any]]:
        """
        Loads the OPSSAT-AD CSV dataset and translates every segment into
        a 7-D State Vector S_t record.

        Returns:
            List of parsed dictionary records per temporal segment t.
        """
        logger.info("[OPSSATParser] Loading CSV dataset: %s", self.csv_path)
        df = pd.read_csv(self.csv_path)
        self._raw_df = df

        # Validate required columns
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"[OPSSATParser] Missing required columns in dataset: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        logger.info(
            "[OPSSATParser] Ingesting %d segments across %d channels...",
            len(df), df["channel"].nunique(),
        )

        self.parsed_records = []

        for idx, row in df.iterrows():
            segment_idx = int(row.get("segment", idx))
            anomaly = int(row["anomaly"])
            channel = str(row["channel"])
            diff_peaks = float(row["diff_peaks"])
            var_div_duration = float(row["var_div_duration"])
            std_val = float(row["std"])
            gaps_squared = float(row["gaps_squared"])

            s_t = self.compute_state_vector(
                anomaly=anomaly,
                diff_peaks=diff_peaks,
                var_div_duration=var_div_duration,
                std_val=std_val,
                gaps_squared=gaps_squared,
                channel=channel,
                segment_index=segment_idx,
            )

            record = {
                "segment_index": segment_idx,
                "row_index": int(idx),
                "channel": channel,
                "anomaly": anomaly,
                "diff_peaks": diff_peaks,
                "var_div_duration": var_div_duration,
                "std": std_val,
                "gaps_squared": gaps_squared,
                "C_t": s_t.C,
                "R_t": s_t.R,
                "T_t": s_t.T,
                "Q_t": s_t.Q,
                "M_t": s_t.M,
                "E_t": s_t.E,
                "A_t": s_t.A,
                "S_t": s_t,
                "active_threats": list(s_t.active_threats),
                "pace_state": s_t.pace_state.value,
            }
            self.parsed_records.append(record)

        self._is_loaded = True
        logger.info(
            "[OPSSATParser] Successfully parsed and translated %d segments.",
            len(self.parsed_records),
        )
        return self.parsed_records

    def get_segment(self, t: int) -> Dict[str, Any]:
        """Returns parsed telemetry record for segment index t (0-based row order)."""
        if not self._is_loaded:
            self.load_and_parse()
        if 0 <= t < len(self.parsed_records):
            return self.parsed_records[t]
        raise IndexError(
            f"Segment index {t} out of range [0, {len(self.parsed_records) - 1}]."
        )

    def get_state_vector(self, t: int) -> StateVectorData:
        """Returns strongly-typed StateVectorData S_t for segment t."""
        return self.get_segment(t)["S_t"]

    def to_dataframe(self) -> pd.DataFrame:
        """Returns full translated dataset as a clean pandas DataFrame."""
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

    def get_raw_dataframe(self) -> pd.DataFrame:
        """Returns the original unmodified CSV as a pandas DataFrame."""
        if self._raw_df is None:
            self.load_and_parse()
        return self._raw_df

    def get_summary(self) -> Dict[str, Any]:
        """Computes executive summary statistics across the parsed dataset."""
        if not self._is_loaded:
            self.load_and_parse()

        a_vals = [r["A_t"] for r in self.parsed_records]
        t_vals = [r["T_t"] for r in self.parsed_records]
        e_vals = [r["E_t"] for r in self.parsed_records]
        c_vals = [r["C_t"] for r in self.parsed_records]
        anomaly_count = sum(1 for r in self.parsed_records if r["anomaly"] == 1)
        threat_count = sum(1 for r in self.parsed_records if r["active_threats"])
        channels = list(set(r["channel"] for r in self.parsed_records))

        return {
            "total_segments": len(self.parsed_records),
            "anomaly_segments": anomaly_count,
            "nominal_segments": len(self.parsed_records) - anomaly_count,
            "unique_channels": channels,
            "num_channels": len(channels),
            "min_A_t": round(min(a_vals), 4),
            "mean_A_t": round(sum(a_vals) / len(a_vals), 4),
            "min_T_t": round(min(t_vals), 4),
            "mean_T_t": round(sum(t_vals) / len(t_vals), 4),
            "min_E_t": round(min(e_vals), 4),
            "mean_E_t": round(sum(e_vals) / len(e_vals), 4),
            "min_C_t": round(min(c_vals), 4),
            "mean_C_t": round(sum(c_vals) / len(c_vals), 4),
            "active_threat_segments": threat_count,
        }

    def get_segments_by_channel(self, channel: str) -> List[Dict[str, Any]]:
        """Filters parsed records by subsystem channel name."""
        if not self._is_loaded:
            self.load_and_parse()
        return [r for r in self.parsed_records if r["channel"] == channel]

    def get_anomaly_segments(self) -> List[Dict[str, Any]]:
        """Returns only segments where anomaly == 1 (malicious injection active)."""
        if not self._is_loaded:
            self.load_and_parse()
        return [r for r in self.parsed_records if r["anomaly"] == 1]

    def get_nominal_segments(self) -> List[Dict[str, Any]]:
        """Returns only segments where anomaly == 0 (nominal operations)."""
        if not self._is_loaded:
            self.load_and_parse()
        return [r for r in self.parsed_records if r["anomaly"] == 0]
