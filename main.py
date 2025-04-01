from typing import Sequence
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def principal_stress(sigma: Sequence[float]) -> tuple[float, float]:
    sigma_array = np.array(sigma)

    sigma_t = np.array(
        [[sigma_array[0], sigma_array[2]], [sigma_array[2], sigma_array[1]]]
    )

    eigenvalues, eigenvectors = np.linalg.eig(sigma_t)
    eigenvalues.sort()

    return float(eigenvalues[0]), float(eigenvalues[1])


def main(sigma_0: list[float]) -> float:
    if len(sigma_0) != 3:
        raise ValueError(
            f"Wrong input for sigma_0, length should be 3, was {len(sigma_0)}"
        )

    sigma_vector = np.array(sigma_0)

    # Constants
    f_cp = 35.0
    E_c = 35.0e3
    f_sy = 500.0
    E_s = 210.0e3
    rho_x = 0.01
    rho_y = 0.01
    G_c = 0.5 * E_c
    k_c = 0.1
    k_s = 0.1

    M_c = np.array(
        [
            [np.sqrt(1 / E_c), np.sqrt(1 / E_c)],
            [0, np.sqrt(1 / (k_c * E_c)) - np.sqrt(1 / E_c)],
        ]
    )

    M_cg = np.array(
        [
            [np.sqrt(1 / G_c), np.sqrt(1 / G_c)],
            [0, np.sqrt(1 / (k_c * G_c)) - np.sqrt(1 / G_c)],
        ]
    )

    M_sx = np.array(
        [
            [np.sqrt(rho_x / E_s), np.sqrt(rho_x / E_s)],
            [0, np.sqrt(rho_x / (k_s * E_s)) - np.sqrt(rho_x / E_s)],
        ]
    )

    M_sy = np.array(
        [
            [np.sqrt(rho_y / E_s), np.sqrt(rho_y / E_s)],
            [0, np.sqrt(rho_y / (k_s * E_s)) - np.sqrt(rho_y / E_s)],
        ]
    )

    # Variables

    ## Total stresses
    st = cp.Variable(3, name="st")

    ## Concrete stresses
    sc = cp.Variable(3, name="sc")
    scxx = sc[0]
    scyy = sc[1]
    scxy = sc[2]

    ## Steel stresses
    ss = cp.Variable(3, name="ss")
    ssxx = ss[0]
    ssyy = ss[1]
    ssxy = ss[2]

    ### Concrete stresses constitutive decomposition
    scxx_lh = cp.Variable(2)
    scxx_l = scxx_lh[0]
    scxx_h = scxx_lh[1]

    scyy_lh = cp.Variable(2)
    scyy_l = scyy_lh[0]
    scyy_h = scyy_lh[1]

    scxy_lh = cp.Variable(2)
    scxy_l = scxy_lh[0]
    scxy_h = scxy_lh[1]

    abs_scxx_lh = cp.Variable(2)
    abs_scyy_lh = cp.Variable(2)
    abs_scxy_lh = cp.Variable(2)

    ## Steel stresses constitutive decomposition
    ssxx_lh = cp.Variable(2)
    ssxx_l = ssxx_lh[0]
    ssxx_h = ssxx_lh[1]

    ssyy_lh = cp.Variable(2)
    ssyy_l = ssyy_lh[0]
    ssyy_h = ssyy_lh[1]

    # ssxy_lh = cp.Variable(2)
    # ssxy_l = ssxy_lh[0]
    # ssxy_h = ssxy_lh[1]

    abs_ssxx_lh = cp.Variable(2)
    abs_ssyy_lh = cp.Variable(2)

    ## Auxiliary
    soc_1 = cp.Variable(3)
    r = cp.Variable()
    sigma_cd = cp.Variable()
    soc_2 = cp.Variable(3)
    r_l = cp.Variable()
    sigma_cd_l = cp.Variable()
    alpha = cp.Variable()

    beta_1 = cp.Variable()
    beta_2 = cp.Variable(2)
    beta_3 = cp.Variable(2)
    beta_4 = cp.Variable(2)
    beta_5 = cp.Variable(2)
    beta_6 = cp.Variable(2)

    # constraints
    constraints: Sequence[cp.Constraint] = []

    ## Stress decomposition
    constraints.append(st == sc + cp.multiply(ss, np.array([rho_x, rho_y, 0])))

    ## Stress equilibrium
    constraints.append(st == sigma_vector)

    ## elastic/hardening stress decomposition
    constraints.append(scxx == scxx_l + scxx_h)
    constraints.append(scyy == scyy_l + scyy_h)
    constraints.append(scxy == scxy_l + scxy_h)
    constraints.append(ssxx == ssxx_l + ssxx_h)
    constraints.append(ssyy == ssyy_l + ssyy_h)

    ## Absolute value constraints
    constraints.append(cp.abs(scxx_lh) <= abs_scxx_lh)
    constraints.append(cp.abs(scyy_lh) <= abs_scyy_lh)
    constraints.append(cp.abs(scxy_lh) <= abs_scxy_lh)

    constraints.append(cp.abs(ssxx_lh) <= abs_ssxx_lh)
    constraints.append(cp.abs(ssyy_lh) <= abs_ssyy_lh)

    ## Tensile cutoff
    constraints.append(r <= -0.5 * (scxx + scyy))
    constraints.append(sigma_cd == 0.5 * (scxx - scyy))
    constraints.append(soc_1[0] == r)
    constraints.append(soc_1[1] == sigma_cd)
    constraints.append(soc_1[2] == scxy)

    constraints.append(
        cp.SOC(
            t=np.array([1, 0, 0]) @ soc_1,
            X=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]) @ soc_1,
        )
    )

    ## plastic concrete strength
    constraints.append(r_l <= f_cp + 0.5 * (scxx_l + scyy_l))
    constraints.append(sigma_cd_l == 0.5 * (scxx_l - scyy_l))

    constraints.append(soc_2[0] == r_l)
    constraints.append(soc_2[1] == sigma_cd_l)
    constraints.append(soc_2[2] == scxy_l)
    constraints.append(
        cp.SOC(
            t=np.array([1, 0, 0]) @ soc_2,
            X=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]) @ soc_2,
        )
    )

    ## Reinforcement yield
    constraints.append(ssxy == 0)

    ### lower limit
    constraints.append(-f_sy <= ssxx_l)
    constraints.append(-f_sy <= ssyy_l)
    ### upper limit
    constraints.append(ssxx_l <= f_sy)
    constraints.append(ssyy_l <= f_sy)

    ## Complimentary energy
    constraints.append(beta_1 == 1)
    constraints.append(beta_2 == M_c @ abs_scxx_lh)
    constraints.append(beta_3 == M_c @ abs_scyy_lh)
    constraints.append(beta_4 == M_cg @ abs_scxy_lh)
    constraints.append(beta_5 == M_sx @ abs_ssxx_lh)
    constraints.append(beta_6 == M_sy @ abs_ssyy_lh)
    beta_all = cp.hstack([beta_2, beta_3, beta_4, beta_5, beta_6])
    constraints.append(alpha >= cp.quad_over_lin(beta_all, beta_1 * 2))

    # Objective
    objective = cp.Minimize(alpha)

    # Create and solve problem
    prob = cp.Problem(
        objective=objective,
        constraints=constraints,
    )
    prob.solve(verbose=False, solver="MOSEK")

    # Print result.
    print("The optimal value is", prob.value)
    print(f"st     = {st.value}")
    print(f"sc     = {sc.value}")
    print(f"ss*rho = {ss.value * [rho_x, rho_y, 0]}\n")

    print(f"st1, st2 = {principal_stress(st.value)}")
    print(f"sc1, sc2 = {principal_stress(sc.value)}")
    print(f"ss1, ss2 = {principal_stress(ss.value)}\n")

    sc_l = [float(scxx_l.value), float(scyy_l.value), float(scxy_l.value)]
    print(f"sc_l = {sc_l}")
    print(f"sc_l1, sc_l2 = {principal_stress(sc_l)}\n")

    sc_h = [float(scxx_h.value), float(scyy_h.value), float(scxy_h.value)]
    print(f"sc_h = {sc_h}")
    print(f"sc_h1, sc_h2 = {principal_stress(sc_h)}\n")

    print(f"scxx_lh = {scxx_lh.value}")
    print(f"scyy_lh = {scyy_lh.value}")
    print(f"scxy_lh = {scxy_lh.value}")

    print(f"ssxx_lh = {ssxx_lh.value}")
    print(f"ssyy_lh = {ssyy_lh.value}")

    print(f"beta_all = {beta_all.value}")

    return float(prob.value)


if __name__ == "__main__":
    x: list[float] = []
    y: list[float] = []

    for i in range(100):
        sigma_0: list[float] = [-i, -i, -i]
        # sigma_0: list[float] = [0.5 * -i, 0.5 * -i, -0.1 * i]

        obj = main(sigma_0=sigma_0)

        x.append(i)
        y.append(obj)

    plt.plot(x, y)
    plt.show()
