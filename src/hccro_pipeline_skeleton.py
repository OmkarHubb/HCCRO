"""
HCCRO Pipeline Skeleton — 8-Stage Cognitive Architecture Pipeline Execution Core
"""

from src.core.orchestrator import HCCROOrchestrator
from src.data.opsat_parser import RealWorldOPSSATParser

__all__ = ["HCCROOrchestrator", "RealWorldOPSSATParser"]
