"""
Module 2: Multi-Objective Pareto Frontier Generator
===================================================
Generates non-dominated decision vector sets using scalarization sweeps
(weighted sum and epsilon-constraint sweeps) balancing Resilience, Mission, and Energy.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from src.core.interfaces import StateVectorData
from src.optimization.constrained_solver import ConstrainedResilienceSolver, OptimizationResult
from src.utils.logger import get_logger

logger = get_logger("Optimization.ParetoFrontier")


class ParetoFrontierGenerator:
    """
    Generates Pareto optimal non-dominated frontiers across competing objectives:
    - Objective 1: Resilience R(x)
    - Objective 2: Mission Throughput M(x)
    - Objective 3: Energy Efficiency E_eff(x) = 1.0 - E_draw(x)
    """

    def __init__(self, solver: Optional[ConstrainedResilienceSolver] = None):
        self.solver = solver or ConstrainedResilienceSolver()

    def generate_frontier(
        self,
        s_t: StateVectorData,
        num_samples: int = 10,
        e_avail: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sweeps weight allocations across R, M, and E to compute the Pareto non-dominated set.
        
        Args:
            s_t: Current satellite StateVectorData
            num_samples: Number of weight scalarization samples
            e_avail: Available energy limit
            
        Returns:
            List of non-dominated Pareto solution dicts containing weights, x_opt, and objective values.
        """
        raw_candidates = []
        
        # Grid sweep of weight space: alpha (Resilience) vs beta (Mission) vs delta (Comm)
        alphas = np.linspace(0.1, 0.8, num_samples)
        
        for a in alphas:
            b = (1.0 - a) * 0.6
            c = (1.0 - a) * 0.4
            
            weights = {
                "alpha_R": round(float(a), 4),
                "beta_M": round(float(b), 4),
                "gamma_T": 0.1,
                "delta_C": round(float(c), 4),
                "lambda_H": 0.1,
            }
            
            self.solver.set_weights(weights)
            res = self.solver.solve(s_t, e_avail=e_avail)
            
            if res.success:
                x = res.x_opt
                # Calculate objective tuple: (R, M, E_eff)
                p_comm, p_tx, f_cpu, m_mem = x["p_comm"], x["p_tx"], x["f_cpu"], x["m_mem"]
                r_obj = s_t.R * (0.5 + 0.5 * f_cpu)
                m_obj = s_t.M * (0.4 * f_cpu + 0.6 * p_tx)
                e_draw = 0.35 * p_comm + 0.30 * p_tx + 0.25 * f_cpu + 0.10 * m_mem
                e_eff = 1.0 - e_draw

                raw_candidates.append({
                    "weights": weights,
                    "x_opt": x,
                    "utility": res.utility_value,
                    "objectives": {
                        "R": round(float(r_obj), 4),
                        "M": round(float(m_obj), 4),
                        "E_eff": round(float(e_eff), 4),
                    }
                })

        # Pareto Dominance Filtering
        pareto_set = self._filter_dominated(raw_candidates)
        logger.info("[ParetoFrontier] Generated %d candidate points, filtered to %d non-dominated solutions.",
                    len(raw_candidates), len(pareto_set))
        return pareto_set

    def _filter_dominated(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters out dominated candidate points."""
        pareto_set = []
        
        for i, c1 in enumerate(candidates):
            dominated = False
            o1 = c1["objectives"]
            
            for j, c2 in enumerate(candidates):
                if i == j:
                    continue
                o2 = c2["objectives"]
                # c2 dominates c1 if c2 is >= c1 in all objectives AND strictly > in at least one
                if (o2["R"] >= o1["R"] and o2["M"] >= o1["M"] and o2["E_eff"] >= o1["E_eff"]) and \
                   (o2["R"] > o1["R"] or o2["M"] > o1["M"] or o2["E_eff"] > o1["E_eff"]):
                    dominated = True
                    break
            
            if not dominated:
                pareto_set.append(c1)

        return pareto_set
