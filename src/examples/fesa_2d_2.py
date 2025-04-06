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

    return float(eigenvalues[1]), float(eigenvalues[0])


def setup_principal_stress_constraints(
    s_vector: cp.Variable, ps_i: cp.Variable, ps_ii: cp.Variable
) -> list[cp.Constraint]:
    constraints: list[cp.Constraint] = []

    c = cp.Variable()
    r = cp.Variable()
    soc_vars = cp.Variable(3)

    constraints.append(ps_i == c + r)
    constraints.append(ps_ii == c - r)
    constraints.append(c == 0.5 * (s_vector[0] + s_vector[1]))

    constraints.append(soc_vars[0] == r)
    constraints.append(soc_vars[1] == 0.5 * (s_vector[0] - s_vector[1]))
    constraints.append(soc_vars[2] == s_vector[2])
    constraints.append(
        cp.SOC(
            np.array([1, 0, 0]) @ soc_vars,
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]) @ soc_vars,
        )
    )
    return constraints


def main(sigma_0: list[float]) -> tuple:
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
    rho_x = 0.05
    rho_y = 0.05
    k_c = 0.01
    k_s = 0.01

    M_c = np.array(
        [
            [np.sqrt(1 / E_c), np.sqrt(1 / E_c)],
            [0, np.sqrt(1 / (k_c * E_c)) - np.sqrt(1 / E_c)],
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

    ## Steel stresses
    ss = cp.Variable(3, name="ss")

    ### Concrete stresses constitutive decomposition
    sc_l = cp.Variable(3, name="sc_l")
    sc_h = cp.Variable(3, name="sc_h")

    ## Steel stresses constitutive decomposition
    ss_l = cp.Variable(3, name="ss_l")
    ss_h = cp.Variable(3, name="ss_h")

    abs_ss_l = cp.Variable(3)
    abs_ss_h = cp.Variable(3)

    ## Concrete principal stresses
    ps_l_i = cp.Variable(name="ps_l_i")
    ps_l_ii = cp.Variable(name="ps_l_ii")
    ps_h_i = cp.Variable(name="ps_h_i")
    ps_h_ii = cp.Variable(name="ps_h_ii")

    ## Auxiliary
    alpha = cp.Variable()

    beta_2 = cp.Variable(2)
    beta_3 = cp.Variable(2)
    beta_4 = cp.Variable(2)
    beta_5 = cp.Variable(2)

    # constraints
    constraints: Sequence[cp.Constraint] = []

    ## Stress decomposition
    constraints.append(st == sc + cp.multiply(ss, np.array([rho_x, rho_y, 0])))

    ## Stress equilibrium
    stress_equilibrium = st == sigma_vector
    constraints.append(stress_equilibrium)

    ## elastic/hardening stress decomposition
    constraints.append(sc == sc_l + sc_h)
    constraints.append(ss == ss_l + ss_h)

    ## Absolute steel stresses
    constraints.append(abs_ss_l >= cp.abs(ss_l))
    constraints.append(abs_ss_h >= cp.abs(ss_h))

    ## Concrete
    constraints.extend(setup_principal_stress_constraints(sc_l, ps_l_i, ps_l_ii))
    constraints.extend(setup_principal_stress_constraints(sc_h, ps_h_i, ps_h_ii))

    ### Limit stress of lower to yielding
    constraints.append(-ps_l_ii <= f_cp)
    constraints.append(ps_l_i <= 0)
    constraints.append(ps_h_i <= 0)

    ## Reinforcement yield
    constraints.append(ss[2] == 0)

    ### lower limit
    constraints.append(-f_sy <= ss_l)
    ### upper limit
    constraints.append(ss_l <= f_sy)

    ## Complimentary energy
    constraints.append(beta_2 == M_c @ cp.hstack([-ps_l_i, -ps_h_i]))
    constraints.append(beta_3 == M_c @ cp.hstack([-ps_l_ii, -ps_h_ii]))
    constraints.append(beta_4 == M_sx @ cp.hstack([abs_ss_l[0], abs_ss_h[0]]))
    constraints.append(beta_5 == M_sy @ cp.hstack([abs_ss_l[1], abs_ss_h[1]]))
    beta_all = cp.hstack([beta_2, beta_3, beta_4, beta_5])
    constraints.append(alpha >= cp.quad_over_lin(beta_all, 2))

    # Objective
    objective = cp.Minimize(alpha)

    # Create and solve problem
    prob = cp.Problem(
        objective=objective,
        constraints=constraints,
    )
    prob.solve(verbose=False, solver="CLARABEL")

    # Print result.
    print("The optimal value is", prob.value)
    print(f"st     = {st.value}")
    print(f"sc     = {sc.value}")
    print(f"ss*rho = {ss.value * [rho_x, rho_y, 0]}\n")

    print(f"st1, st2 = {principal_stress(st.value)}")
    print(f"sc1, sc2 = {principal_stress(sc.value)}")
    print(f"ss1, ss2 = {principal_stress(ss.value)}\n")

    print(f"sc_l = {sc_l.value}")
    print(f"sc_l1, sc_l2 = {float(ps_l_i.value), float(ps_l_ii.value)}")
    print(f"sc_l1, sc_l2 = {principal_stress(sc_l.value)}\n")

    print(f"sc_h = {sc_h.value}")
    print(f"sc_h1, sc_h2 = {float(ps_h_i.value), float(ps_h_ii.value)}")
    print(f"sc_h1, sc_h2 = {principal_stress(sc_h.value)}\n")

    print(f"ss_l = {ss_l.value}")
    print(f"ss_l1, ss_l2 = {principal_stress(ss_l.value)}\n")

    print(f"ss_h = {ss_h.value}")
    print(f"ss_h1, ss_h2 = {principal_stress(ss_h.value)}\n")

    print(f"beta_all = {beta_all.value}")

    sc_I = principal_stress(sc_l.value)[0] + principal_stress(sc_h.value)[0]
    ec_I = principal_stress(sc_l.value)[0] / E_c + principal_stress(sc_h.value)[0] / (
        E_c * k_c
    )

    sc_II = principal_stress(sc_l.value)[1] + principal_stress(sc_h.value)[1]
    ec_II = principal_stress(sc_l.value)[1] / E_c + principal_stress(sc_h.value)[1] / (
        E_c * k_c
    )

    ss_xx = float(ss[0].value)
    es_xx = float(ss_l[0].value / E_s + ss_h[0].value / (k_s * E_s))

    ss_yy = float(ss[1].value)
    es_yy = float(ss_l[1].value / E_s + ss_h[1].value / (k_s * E_s))

    return (sc_I, ec_I, sc_II, ec_II, ss_xx, es_xx, ss_yy, es_yy)


if __name__ == "__main__":
    # sigma_0: list[float] = [-20, 0, 0]
    # # sigma_0: list[float] = [0.5 * -i, 0.5 * -i, -0.1 * i]

    # (sig_I, eps_I, sig_II, eps_II, ss_xx, es_xx, ss_yy, es_yy) = main(sigma_0=sigma_0)

    sig_Is = []
    eps_Is = []
    eps_IIs = []
    sig_IIs = []
    ss_xxs = []
    es_xxs = []
    ss_yys = []
    es_yys = []

    for i in np.linspace(start=-80, stop=30, num=200):
        sigma_0: list[float] = [i, i, 0.2 * i]
        # sigma_0: list[float] = [0.5 * -i, 0.5 * -i, -0.1 * i]

        (sig_I, eps_I, sig_II, eps_II, ss_xx, es_xx, ss_yy, es_yy) = main(
            sigma_0=sigma_0
        )

        sig_Is.append(sig_I)
        eps_Is.append(eps_I)
        eps_IIs.append(eps_II)
        sig_IIs.append(sig_II)
        ss_xxs.append(ss_xx)
        es_xxs.append(es_xx)
        ss_yys.append(ss_yy)
        es_yys.append(es_yy)

    fig, axs = plt.subplots(2, 1, layout="constrained")

    vh_line_dict = {"ls": ":", "lw": 0.5}
    coord_line_dict = {"ls": "-", "lw": 0.75, "color": "darkgrey"}

    axs[0].axvline(0, **coord_line_dict)
    axs[0].axhline(0, **coord_line_dict)
    axs[0].axvline(-35 / 35e3, **vh_line_dict)
    axs[0].axhline(-35, **vh_line_dict)

    axs[1].axvline(0, **coord_line_dict)
    axs[1].axhline(0, **coord_line_dict)
    axs[1].axvline(-500 / 210e3, **vh_line_dict)
    axs[1].axvline(500 / 210e3, **vh_line_dict)
    axs[1].axhline(-500, **vh_line_dict)
    axs[1].axhline(500, **vh_line_dict)

    axs[0].set_title("Concrete work curve")
    axs[0].plot(eps_Is, sig_Is)
    axs[0].plot(eps_IIs, sig_IIs)
    axs[1].set_title("Steel work curve")
    axs[1].plot(es_xxs, ss_xxs)
    axs[1].plot(es_yys, ss_yys)

    plt.show()
