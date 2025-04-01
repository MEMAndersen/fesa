"""
Example of Reinforced Concrete material point in 2D with orthogonal reinforcement in two directions (x-y).
"""

# Import packages.
from typing import Sequence
import cvxpy as cp
from cvxpy.constraints.zero import Equality
import numpy as np


def main(sigma: list[float], sigma_0: list[float]) -> None:
    if len(sigma) != 3:
        raise ValueError(f"Wrong input for sigma, length should be 3, was {len(sigma)}")

    if len(sigma_0) != 3:
        raise ValueError(
            f"Wrong input for sigma_0, length should be 3, was {len(sigma_0)}"
        )

    sigma_load = np.array(sigma)

    sigma_0_load = np.array(sigma_0)

    # Define variables
    k = 4.0
    f_t_eff = 0.0  # MPa
    f_c_eff = 20.0  # MPa
    phi_x = 5  # MPa
    phi_y = 5  # MPa

    ## Stresses
    s_c = cp.Variable((2, 2), symmetric=True, name="S_c")
    s_s = cp.Variable((2, 2), symmetric=True, name="S_s")
    s_t = cp.Variable((2, 2), symmetric=True, name="S_t")

    ## Auxiliary
    s_ii = cp.Variable(1, name="s_ii")
    s_i = cp.Variable(1, name="s_i")
    c = cp.Variable(1, name="C")
    r = cp.Variable(1, name="R")
    load_factor = cp.Variable(1, name="load_factor")
    soc_vars = cp.Variable(3)

    # Constraints
    stress_equality = load_factor * sigma_load + sigma_0_load == s_t
    stress_decomposition_constraint = s_t == s_c + s_s

    ## Concrete
    s_i_eq = s_i == c + r
    s_ii_eq = s_ii == c - r
    c_eq = c == 0.5 * (s_c[0, 0] + s_c[1, 1])
    soc_eqs: Sequence[Equality | cp.SOC] = [
        soc_vars[0] == r,
        soc_vars[1] == 0.5 * (s_c[0, 0] - s_c[1, 1]),
        soc_vars[2] == s_c[0, 1],
        cp.SOC(
            np.array([1, 0, 0]) @ soc_vars,
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]) @ soc_vars,
        ),
    ]

    mohr_1 = c + r <= f_t_eff
    mohr_2 = k * (c + r) - (c - r) <= f_c_eff
    mohr_3 = -(c - r) <= f_c_eff

    ## Steel
    s_lin_ineq_low = np.zeros((2, 2)) <= s_s
    s_lin_ineq_high = s_s <= np.diag((phi_x, phi_y))

    # Objective
    objective = cp.Maximize(load_factor)

    # Create and solve problem
    constraints: Sequence[cp.Constraint] = [
        stress_equality,
        stress_decomposition_constraint,
        s_i_eq,
        s_ii_eq,
        c_eq,
        mohr_1,
        mohr_2,
        mohr_3,
        s_lin_ineq_low,
        s_lin_ineq_high,
    ] + soc_eqs

    prob = cp.Problem(
        objective=objective,
        constraints=constraints,
    )
    prob.solve(verbose=True, solver="MOSEK")

    # Print result.
    print("The optimal value is", prob.value)
    print(f"S_t = \n{s_t.value}")
    print(f"S_c = \n{s_c.value}")
    print(f"S_s = \n{s_s.value}")
    print(load_factor.value * sigma_load + sigma_0_load, "\n=\n", s_t.value)
    print(f"sii = \n{s_ii.value}")
    print(f"si = \n{s_i.value}")


if __name__ == "__main__":
    sigma: list[float] = [-1, 0, 0, 0]
    sigma_0: list[float] = [0, 0, 0]

    main(sigma=sigma, sigma_0=sigma_0)
