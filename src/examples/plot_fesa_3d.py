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
reinforcement_mat = ReinforcementMaterial(fY=200, E=200e3, kappa=0.05, gamma=1)
reinforcement = [
    Reinforcement(direction=(1, 0, 0), rho=0.01, material=reinforcement_mat),
    Reinforcement(direction=(0, 1, 0), rho=0.01, material=reinforcement_mat),
    Reinforcement(direction=(0, 0, 1), rho=0.01, material=reinforcement_mat),
]

mp_fesa = MaterialPointFESA3D(
    sigma=StressTensor(),
    concrete_material=concrete_mat,
    reinforcement=reinforcement,
)
mp_fesa.setup_variables()
mp_fesa.define_constraints()

ps_is_l = []
ps_iis_l = []
ps_iiis_l = []

ps_is_h = []
ps_iis_h = []
ps_iiis_h = []


for i in tqdm(range(1000)):
    sigma_ext_array: list[float] = (np.random.rand(6) * 70 - 35).tolist()  # type: ignore
    sigma_ext = list_2_tensor(sigma_ext_array)

    # Create and solve problem
    total_equilibrium = sigma_ext == mp_fesa.sigma.tensor
    prob = cp.Problem(
        objective=cp.Minimize(mp_fesa.objective),
        constraints=mp_fesa.constraint_list + [total_equilibrium],
    )
    prob.solve(verbose=False, solver="MOSEK")

    ps_is_l.append(mp_fesa.aux_variable_dict["ps_l"][0].value)
    ps_iis_l.append(mp_fesa.aux_variable_dict["ps_l"][1].value)
    ps_iiis_l.append(mp_fesa.aux_variable_dict["ps_l"][2].value)

    ps_is_h.append(mp_fesa.aux_variable_dict["ps_h"][0].value)
    ps_iis_h.append(mp_fesa.aux_variable_dict["ps_h"][1].value)
    ps_iiis_h.append(mp_fesa.aux_variable_dict["ps_h"][2].value)

ps_is_l = np.array(ps_is_l)
ps_iis_l = np.array(ps_iis_l)
ps_iiis_l = np.array(ps_iiis_l)
ps_is_h = np.array(ps_is_h)
ps_iis_h = np.array(ps_iis_h)
ps_iiis_h = np.array(ps_iiis_h)

fig, axs = plt.subplots(2, 1, layout="constrained")

vh_line_dict = {"ls": ":", "lw": 0.5}
coord_line_dict = {"ls": "-", "lw": 0.75, "color": "darkgrey"}

axs[0].axvline(0, **coord_line_dict)
axs[0].axhline(0, **coord_line_dict)
axs[0].axvline(-fyc / E_c, **vh_line_dict)
axs[0].axhline(-fyc, **vh_line_dict)

axs[1].axvline(0, **coord_line_dict)
axs[1].axhline(0, **coord_line_dict)
axs[1].axvline(-500 / 210e3, **vh_line_dict)
axs[1].axvline(500 / 210e3, **vh_line_dict)
axs[1].axhline(-500, **vh_line_dict)
axs[1].axhline(500, **vh_line_dict)

axs[0].set_title("Concrete work curve")
axs[0].plot(ps_is_l / E_c + ps_is_h / (E_c * kappa_c), ps_is_l + ps_is_h, "b.")
axs[0].plot(ps_iis_l / E_c + ps_iis_h / (E_c * kappa_c), ps_iis_l + ps_iis_h, "r.")
axs[0].plot(ps_iiis_l / E_c + ps_iiis_h / (E_c * kappa_c), ps_iiis_l + ps_iiis_h, "k.")
axs[1].set_title("Steel work curve")
# axs[1].plot(es_xxs, ss_xxs)
# axs[1].plot(es_yys, ss_yys)

plt.show()
