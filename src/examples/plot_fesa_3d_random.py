from matplotlib import pyplot as plt
import numpy as np
import cvxpy as cp
from tqdm import tqdm
from fesa_3d import (
    ConcreteMaterial,
    MaterialPointFESA3D,
    Reinforcement,
    ReinforcementMaterial,
    StressTensor,
    list_2_tensor,
)

fyc = 35
E_c = 50e3
kappa_c = 0.05

concrete_mat = ConcreteMaterial(fY=fyc, E=E_c, kappa=kappa_c, gamma=1, nu=1)
reinforcement_mat = ReinforcementMaterial(fY=500, E=200e3, kappa=0.05, gamma=1)
reinforcement = [
    Reinforcement(direction=(1, 0, 0), rho=0.05, material=reinforcement_mat),
    Reinforcement(direction=(0, 1, 0), rho=0.05, material=reinforcement_mat),
    Reinforcement(direction=(0, 0, 1), rho=0.05, material=reinforcement_mat),
]

mp_fesa = MaterialPointFESA3D(
    sigma=StressTensor(),
    concrete_material=concrete_mat,
    reinforcement=reinforcement,
)
mp_fesa.setup_variables()
mp_fesa.define_constraints()

nsr: int = len(reinforcement)

ps_is_l = []
ps_iis_l = []
ps_iiis_l = []

ps_is_h = []
ps_iis_h = []
ps_iiis_h = []

ss_ls = [[] for _ in range(nsr)]
ss_hs = [[] for _ in range(nsr)]

for _ in tqdm(range(10000)):
    sigma_ext_array: list[float] = (np.random.rand(6) * 70 - 35).tolist()  # type: ignore
    sigma_ext = list_2_tensor(sigma_ext_array)

    # Create and solve problem
    total_equilibrium = sigma_ext == mp_fesa.sigma.tensor
    prob = cp.Problem(
        objective=cp.Minimize(mp_fesa.objective),
        constraints=mp_fesa.constraint_list + [total_equilibrium],
    )
    prob.solve(
        verbose=False,
        # solver="CLARABEL",
    )

    ps_is_l.append(mp_fesa.aux_variable_dict["ps_l"][0].value)
    ps_iis_l.append(mp_fesa.aux_variable_dict["ps_l"][1].value)
    ps_iiis_l.append(mp_fesa.aux_variable_dict["ps_l"][2].value)

    ps_is_h.append(mp_fesa.aux_variable_dict["ps_h"][0].value)
    ps_iis_h.append(mp_fesa.aux_variable_dict["ps_h"][1].value)
    ps_iiis_h.append(mp_fesa.aux_variable_dict["ps_h"][2].value)

    for i in range(nsr):
        ss_ls[i].append(mp_fesa.reinforcement_stress_list[i].s_l.value)
        ss_hs[i].append(mp_fesa.reinforcement_stress_list[i].s_h.value)

ps_is_l = np.array(ps_is_l)
ps_iis_l = np.array(ps_iis_l)
ps_iiis_l = np.array(ps_iiis_l)
ps_is_h = np.array(ps_is_h)
ps_iis_h = np.array(ps_iis_h)
ps_iiis_h = np.array(ps_iiis_h)

ps_ls = [ps_is_l, ps_iis_l, ps_iiis_l]
ps_hs = [ps_is_h, ps_iis_h, ps_iiis_h]

ss_ls = [np.array(x) for x in ss_ls]
ss_hs = [np.array(x) for x in ss_hs]

fig, axs = plt.subplots(1, 3, layout="constrained")

vh_line_dict = {"ls": ":", "lw": 0.5}
coord_line_dict = {"ls": "-", "lw": 0.75, "color": "darkgrey"}

for i in range(3):
    axs[i].axvline(0, **coord_line_dict)
    axs[i].axhline(0, **coord_line_dict)
    axs[i].axvline(-fyc / E_c, **vh_line_dict)
    axs[i].axhline(-fyc, **vh_line_dict)

    axs[i].set_title(f"PSC {i + 1}")
    axs[i].plot(
        ps_ls[i] / E_c + ps_hs[i] / (E_c * kappa_c),
        ps_ls[i] + ps_hs[i],
        "b.",
        ms=0.1,
    )

fig, axs = plt.subplots(1, nsr, layout="constrained")

for i, r in enumerate(reinforcement):
    fyd = r.material.fY / r.material.gamma
    Es = r.material.E
    kappa_s = r.material.kappa

    axs[i].set_title("Steel work curve")
    axs[i].axvline(0, **coord_line_dict)
    axs[i].axhline(0, **coord_line_dict)
    axs[i].axvline(-fyd / Es, **vh_line_dict)
    axs[i].axvline(fyd / Es, **vh_line_dict)
    axs[i].axhline(-fyd, **vh_line_dict)
    axs[i].axhline(fyd, **vh_line_dict)

    axs[i].set_title(f"SS {i + 1}")
    axs[i].plot(
        ss_ls[i] / Es + ss_hs[i] / (Es * kappa_s),
        ss_ls[i] + ss_hs[i],
        "b.",
        ms=0.1,
    )

plt.show()
