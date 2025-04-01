"""
Example of Reinforced Concrete material point in 3D with orthogonal reinforcement in three directions (x-y-z).
"""

# Import packages.
import cvxpy as cp
import numpy as np


def main(sigma: list[float], sigma_0: list[float]) -> None:
    if len(sigma) != 6:
        raise ValueError(f"Wrong input for sigma, length should be 6, was {len(sigma)}")

    if len(sigma_0) != 6:
        raise ValueError(
            f"Wrong input for sigma_0, length should be 6, was {len(sigma_0)}"
        )

    sigma_tensor = np.array(
        [
            [sigma[0], sigma[3], sigma[4]],
            [sigma[3], sigma[1], sigma[5]],
            [sigma[4], sigma[5], sigma[2]],
        ]
    )

    sigma_0_tensor = np.array(
        [
            [sigma_0[0], sigma_0[3], sigma_0[4]],
            [sigma_0[3], sigma_0[1], sigma_0[5]],
            [sigma_0[4], sigma_0[5], sigma_0[2]],
        ]
    )

    # Define variables
    k = 4.0
    f_t_eff = 0.0  # MPa
    f_c_eff = 20.0  # MPa
    phi_x = 5  # MPa
    phi_y = 5  # MPa
    phi_z = 5  # MPa

    ## Stresses
    S_c = cp.Variable((3, 3), symmetric=True, name="S_c")
    S_s = cp.Variable((3, 3), symmetric=True, name="S_s")
    S_t = cp.Variable((3, 3), symmetric=True, name="S_t")

    ## Auxiliary
    s3 = cp.Variable(1, name="s3")
    s1 = cp.Variable(1, name="s1")
    load_factor = cp.Variable(1, name="load_factor")

    # Constraints
    stress_equality = load_factor * sigma_tensor + sigma_0_tensor == S_t
    stress_decomposition_constraint = S_t == S_c + S_s

    ## Concrete
    sdp_1 = s1 * np.identity(3) - S_c >> 0
    sdp_2 = s3 * np.identity(3) - S_c << 0
    lin_ineq_1 = s1 <= f_t_eff
    lin_ineq_2 = (k * s1 - s3) <= f_c_eff

    ## Steel
    s_lin_ineq_low = np.zeros((3, 3)) <= S_s
    s_lin_ineq_high = S_s <= np.diag((phi_x, phi_y, phi_z))

    # Objective
    objective = cp.Maximize(load_factor)

    # Create and solve problem
    prob = cp.Problem(
        objective=objective,
        constraints=[
            stress_equality,
            stress_decomposition_constraint,
            sdp_1,
            sdp_2,
            lin_ineq_1,
            lin_ineq_2,
            s_lin_ineq_low,
            s_lin_ineq_high,
        ],
    )
    prob.solve(verbose=True, solver="MOSEK")

    # Print result.
    print("The optimal value is", prob.value)
    print(f"S_t = \n{S_t.value}")
    print(f"S_c = \n{S_c.value}")
    print(f"S_s = \n{S_s.value}")
    print(load_factor.value * sigma_tensor + sigma_0_tensor, "\n=\n", S_t.value)
    print(f"s3 = \n{s3.value}")
    print(f"s1 = \n{s1.value}")


if __name__ == "__main__":
    sigma: list[float] = [-20, -20, -2, 0, 0, 0]
    sigma_0: list[float] = [-10, 0, 0, 0, 0, 0]

    main(sigma=sigma, sigma_0=sigma_0)
