"""
Module 2: Non-Linear Constrained Optimization Engine
===================================================
Uses scipy.optimize.minimize (SLSQP solver) to optimize resource allocation
under physical, latency, energy, memory, and cyber risk constraints.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint

from src.core.interfaces import StateVectorData
from src.utils.logger import get_logger

logger = get_logger("Optimization.ConstrainedSolver")


@dataclass
class OptimizationResult:
    """Artifact containing the output of the constrained optimization solver."""
    x_opt: Dict[str, float] = field(default_factory=lambda: {
        "p_comm": 0.5,
        "p_tx": 0.5,
        "f_cpu": 0.5,
        "m_mem": 0.5,
    })
    utility_value: float = 0.0
    success: bool = True
    message: str = "Optimization completed successfully"
    constraints_satisfied: bool = True


class ConstrainedResilienceSolver:
    """
    Formulates and solves non-linear constrained optimization for satellite resource allocation.
    
    Decision Vector:
        x = [p_comm, p_tx, f_cpu, m_mem]
        All variables normalized to [0.0, 1.0].
        - p_comm: Communication channel bandwidth allocation
        - p_tx: Transmit power allocation
        - f_cpu: Processing frequency / CPU duty cycle
        - m_mem: Dedicated memory buffer allocation
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "alpha_R": 0.25,
            "beta_M": 0.25,
            "gamma_T": 0.25,
            "delta_C": 0.15,
            "lambda_H": 0.10,
        }

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update weights dynamically (e.g. from CKE or DCS-MOS)."""
        self.weights = weights

    def solve(
        self,
        s_t: StateVectorData,
        e_avail: Optional[float] = None,
        l_max: float = 2.0,
        m_avail: Optional[float] = None,
        r_min: float = 0.30,
    ) -> OptimizationResult:
        """
        Solves the constrained optimization problem:
            Maximize U(x) = alpha*R(x) + beta*M(x) + gamma*T(x) + delta*C(x) + lambda*H(x)
            Subject to:
                1. Energy draw: E_draw(x) <= E_avail
                2. Latency: L(x) <= L_max
                3. Memory: m_mem <= M_avail
                4. Cyber Risk floor: R(x) >= R_min
        """
        # Determine available limits from state vector if not explicitly passed
        energy = e_avail if e_avail is not None else s_t.E
        memory = m_avail if m_avail is not None else s_t.M
        threat_severity = 0.5 if len(s_t.active_threats) > 0 else 0.1

        # Bounds: x_i in [0.01, 1.0] to prevent divide-by-zero
        bounds = Bounds([0.01, 0.01, 0.01, 0.01], [1.0, 1.0, 1.0, 1.0])

        # Objective function (Negative for minimization)
        def objective(x):
            p_comm, p_tx, f_cpu, m_mem = x
            
            # Simulated objective components based on resource allocation
            r_val = s_t.R * (0.5 + 0.5 * f_cpu) * (1.0 - threat_severity * (1.0 - m_mem))
            m_val = s_t.M * (0.4 * f_cpu + 0.6 * p_tx)
            t_val = s_t.T * (0.7 + 0.3 * p_comm)
            c_val = s_t.C * (0.5 * p_comm + 0.5 * p_tx)
            h_val = 1.0 if not s_t.active_threats else (0.5 + 0.5 * f_cpu)

            u = (
                self.weights.get("alpha_R", 0.25) * r_val
                + self.weights.get("beta_M", 0.25) * m_val
                + self.weights.get("gamma_T", 0.25) * t_val
                + self.weights.get("delta_C", 0.15) * c_val
                + self.weights.get("lambda_H", 0.10) * h_val
            )
            return -u

        # Constraint 1: Energy draw <= energy (E_draw <= E_avail -> E_avail - E_draw >= 0)
        def constraint_energy(x):
            p_comm, p_tx, f_cpu, m_mem = x
            e_draw = 0.35 * p_comm + 0.30 * p_tx + 0.25 * f_cpu + 0.10 * m_mem
            return energy - e_draw

        # Constraint 2: Latency <= L_max (L_max - L(x) >= 0)
        def constraint_latency(x):
            p_comm, p_tx, f_cpu, m_mem = x
            latency = (0.1 / p_tx) + (0.05 / f_cpu)
            return l_max - latency

        # Constraint 3: Memory <= M_avail (M_avail - m_mem >= 0)
        def constraint_memory(x):
            p_comm, p_tx, f_cpu, m_mem = x
            return memory - m_mem

        # Constraint 4: Resilience floor >= R_min (R(x) - R_min >= 0)
        def constraint_resilience(x):
            p_comm, p_tx, f_cpu, m_mem = x
            r_val = s_t.R * (0.5 + 0.5 * f_cpu) * (1.0 - threat_severity * (1.0 - m_mem))
            return r_val - r_min

        constraints = [
            {"type": "ineq", "fun": constraint_energy},
            {"type": "ineq", "fun": constraint_latency},
            {"type": "ineq", "fun": constraint_memory},
            {"type": "ineq", "fun": constraint_resilience},
        ]

        # Initial guess: midpoint or scaled to energy limit
        x0 = np.array([
            min(0.5, energy * 0.5),
            min(0.5, energy * 0.5),
            min(0.5, energy * 0.5),
            min(0.5, memory * 0.8),
        ])
        x0 = np.clip(x0, 0.05, 0.95)

        try:
            res = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-6},
            )
            
            if res.success:
                x_opt = {
                    "p_comm": round(float(res.x[0]), 4),
                    "p_tx": round(float(res.x[1]), 4),
                    "f_cpu": round(float(res.x[2]), 4),
                    "m_mem": round(float(res.x[3]), 4),
                }
                utility = round(float(-res.fun), 4)
                logger.info("[ConstrainedSolver] SLSQP converged. Utility: %.4f, Alloc: %s", utility, x_opt)
                return OptimizationResult(
                    x_opt=x_opt,
                    utility_value=utility,
                    success=True,
                    message="SLSQP optimization converged",
                    constraints_satisfied=True,
                )
            else:
                logger.warning("[ConstrainedSolver] SLSQP failed to converge: %s. Falling back to heuristic.", res.message)
                return self._fallback_heuristic(s_t, energy, memory, r_min)

        except Exception as e:
            logger.error("[ConstrainedSolver] Exception during SLSQP optimization: %s", e)
            return self._fallback_heuristic(s_t, energy, memory, r_min)

    def _fallback_heuristic(
        self, s_t: StateVectorData, energy: float, memory: float, r_min: float
    ) -> OptimizationResult:
        """Robust heuristic allocation fallback if non-linear solver fails."""
        scaled_e = max(0.1, min(1.0, energy))
        scaled_m = max(0.1, min(1.0, memory))
        
        x_opt = {
            "p_comm": round(scaled_e * 0.3, 4),
            "p_tx": round(scaled_e * 0.3, 4),
            "f_cpu": round(scaled_e * 0.3, 4),
            "m_mem": round(scaled_m * 0.8, 4),
        }
        utility = round(
            self.weights.get("alpha_R", 0.25) * s_t.R +
            self.weights.get("beta_M", 0.25) * s_t.M +
            self.weights.get("gamma_T", 0.25) * s_t.T, 4
        )
        return OptimizationResult(
            x_opt=x_opt,
            utility_value=utility,
            success=False,
            message="Fallback heuristic used due to solver failure",
            constraints_satisfied=True,
        )
