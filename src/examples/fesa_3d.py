from abc import abstractmethod
from typing import Sequence
import numpy as np
from numpy.typing import NDArray
import cvxpy as cp
from pydantic import BaseModel, ConfigDict, Field, field_validator

TOL = 1e-9


def setup_principal_stress_constraints(
    s_tensor: cp.Variable, ps: cp.Variable
) -> list[cp.Constraint]:
    constraints: list[cp.Constraint] = []

    ps_i = ps[0]
    ps_ii = ps[1]
    ps_iii = ps[2]

    constraints.append(ps_i * np.identity(3) - s_tensor >> 0)
    constraints.append(ps_iii * np.identity(3) - s_tensor << 0)
    constraints.append(ps_ii == cp.trace(s_tensor) - ps_i - ps_iii)
    return constraints


def list_2_tensor(sigma: list[float]) -> NDArray:
    return np.array(
        [
            [sigma[0], sigma[3], sigma[4]],
            [sigma[3], sigma[1], sigma[5]],
            [sigma[4], sigma[5], sigma[2]],
        ]
    )


class FESAMaterial(BaseModel):
    fY: float
    "Plastic strength / yield strength in MPa"
    E: float
    "Elastic module of elasticity in MPa"
    kappa: float
    "Hardening parameter E_h = E*kappa  kappa < 1"
    gamma: float
    "Partial safety coefficient"


class ConcreteMaterial(FESAMaterial):
    nu: float
    "Effectiveness factor"
    k: float = 4
    "Friction coefficient"

    def M_matrix(self) -> NDArray:
        E = self.E
        k: float = self.kappa

        return np.array(
            [
                [np.sqrt(1 / E), np.sqrt(1 / E)],
                [0, np.sqrt(1 / (k * E)) - np.sqrt(1 / E)],
            ]
        )


class ReinforcementMaterial(FESAMaterial):
    def M_matrix(self, rho: float) -> NDArray:
        E = self.E
        k: float = self.kappa

        return np.array(
            [
                [np.sqrt(rho / E), np.sqrt(rho / E)],
                [0, np.sqrt(rho / (k * E)) - np.sqrt(rho / E)],
            ]
        )


class Reinforcement(BaseModel):
    direction: tuple[float, float, float]
    rho: float
    material: ReinforcementMaterial

    @field_validator("direction", mode="after")
    @classmethod
    def is_magnitude_1(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        magnitude: float = float(abs(np.linalg.norm(value, ord=2)))

        if abs(1.0 - magnitude) > TOL:
            return (value[0] / magnitude, value[1] / magnitude, value[2] / magnitude)
        return value

    def transformation_matrix(self) -> NDArray:
        ns = np.array([self.direction])
        tm = ns.T @ ns
        return tm


class StressTensor(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tensor: cp.Variable = Field(
        default_factory=lambda: cp.Variable((3, 3), symmetric=True), init=False
    )

    @property
    def sxx(self):
        return self.tensor[0, 0]

    @property
    def syy(self):
        return self.tensor[1, 1]

    @property
    def szz(self):
        return self.tensor[2, 2]

    @property
    def sxy(self):
        return self.tensor[0, 1]

    @property
    def sxz(self):
        return self.tensor[0, 2]

    @property
    def syz(self):
        return self.tensor[1, 2]


class ReinforcementStress(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    s_t: cp.Variable = Field(default_factory=lambda: cp.Variable(), init=False)
    "Linear elastic part"
    s_l: cp.Variable = Field(default_factory=lambda: cp.Variable(), init=False)
    "Linear elastic part"
    s_h: cp.Variable = Field(default_factory=lambda: cp.Variable(), init=False)
    "hardening part"
    s_l_abs: cp.Variable = Field(default_factory=lambda: cp.Variable(), init=False)
    "absolute of linear elastic part"
    s_h_abs: cp.Variable = Field(default_factory=lambda: cp.Variable(), init=False)
    "absolute of hardening part"

    def print_values(self):
        print("st = s_l + s_h")
        print(f"{self.s_t.value} = {self.s_l.value} + {self.s_h.value}")
        print(f"s_l_abs = {self.s_l_abs.value}, s_h_abs = {self.s_h_abs.value} ")


class MaterialPointFESA3D(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sigma: StressTensor
    concrete_material: ConcreteMaterial
    reinforcement: list[Reinforcement]
    stress_point_dict: dict[str, StressTensor] = Field(init=False, default={})
    reinforcement_stress_list: list[ReinforcementStress] = Field(init=False, default=[])
    aux_variable_dict: dict[str, cp.Variable] = Field(init=False, default={})
    complimentary_energy_variable_list: list[cp.Variable] = Field(
        init=False, default=[]
    )
    constraint_list: list[cp.Constraint] = Field(init=False, default=[])
    objective: cp.Variable = Field(init=False, default=cp.Variable())

    def setup_variables(self):
        vd = self.aux_variable_dict

        self._setup_stress_points()
        self._setup_reinforcement_variables()

        ## Concrete principal stresses
        vd["ps_l"] = cp.Variable(3)
        vd["ps_h"] = cp.Variable(3)

    def _setup_stress_points(self) -> None:
        spd = self.stress_point_dict
        ## Concrete stresses
        spd["sc"] = StressTensor()

        ## Steel stresses total
        spd["ss"] = StressTensor()

        ### Concrete stresses constitutive decomposition
        spd["sc_l"] = StressTensor()
        spd["sc_h"] = StressTensor()

    def _setup_reinforcement_variables(self) -> None:
        for _ in range(len(self.reinforcement)):
            self.reinforcement_stress_list.append(ReinforcementStress())

    def _setup_complimentary_energy(self) -> None:
        vd = self.aux_variable_dict
        M = self.concrete_material.M_matrix()

        for i in range(3):
            beta = cp.Variable(2, name=f"beta_c{i}")
            self.complimentary_energy_variable_list.append(beta)
            self.constraint_list.append(
                beta == M @ cp.hstack([-vd["ps_l"][i], -vd["ps_h"][i]])
            )

        for i, (r, rs) in enumerate(
            zip(self.reinforcement, self.reinforcement_stress_list)
        ):
            M = r.material.M_matrix(r.rho)
            beta = cp.Variable(2, name=f"beta_s{i}")
            self.complimentary_energy_variable_list.append(beta)
            self.constraint_list.append(beta == M @ cp.hstack([rs.s_l_abs, rs.s_h_abs]))

        self.constraint_list.append(
            self.objective
            >= cp.quad_over_lin(cp.hstack(self.complimentary_energy_variable_list), 2)
        )

    def define_constraints(self) -> None:
        spd = self.stress_point_dict
        vd = self.aux_variable_dict

        # Stress decomposition
        self.constraint_list.append(
            self.sigma.tensor == spd["sc"].tensor + spd["ss"].tensor
        )

        # Combined reinforcement stress into tensor
        ss_all = []
        for r, rs in zip(self.reinforcement, self.reinforcement_stress_list):
            tm = r.transformation_matrix()
            ss_all.append(rs.s_t * r.rho * tm)
        self.constraint_list.append(spd["ss"].tensor == cp.sum(ss_all))

        # elastic/hardening stress decomposition

        ## Concrete
        self.constraint_list.append(
            spd["sc"].tensor == spd["sc_l"].tensor + spd["sc_h"].tensor
        )

        ## Steel
        for r, rs in zip(self.reinforcement, self.reinforcement_stress_list):
            # Summation
            self.constraint_list.append(rs.s_t == rs.s_l + rs.s_h)
            # Absolute values for complimentary energy
            self.constraint_list.append(cp.abs(rs.s_l) <= rs.s_l_abs)
            self.constraint_list.append(cp.abs(rs.s_h) <= rs.s_h_abs)

        # Principal concrete stresses
        self.constraint_list.extend(
            setup_principal_stress_constraints(spd["sc_l"].tensor, vd["ps_l"])
        )
        self.constraint_list.extend(
            setup_principal_stress_constraints(spd["sc_h"].tensor, vd["ps_h"])
        )

        # Stress limits
        self._stress_limits()

        # Complimentary energy
        self._setup_complimentary_energy()

    def _stress_limits(self):
        ps_l = self.aux_variable_dict["ps_l"]
        ps_h = self.aux_variable_dict["ps_h"]

        # Limit concrete to compression
        self.constraint_list.append(ps_l[0] <= 0)
        self.constraint_list.append(ps_h <= 0)

        # Limit linear concrete stress to fcp_eff
        cm = self.concrete_material
        k = cm.k
        fcp_eff = (cm.fY * cm.nu) / cm.gamma
        self.constraint_list.append(k * ps_l[0] - ps_l[2] <= fcp_eff)

        # limit steel stresses
        for r, rs in zip(self.reinforcement, self.reinforcement_stress_list):
            fyd = r.material.fY / r.material.gamma
            self.constraint_list.append(-fyd <= rs.s_l)
            self.constraint_list.append(rs.s_l <= fyd)


if __name__ == "__main__":
    concrete_mat = ConcreteMaterial(fY=35, E=50e3, kappa=0.05, gamma=1, nu=1)
    reinforcement_mat = ReinforcementMaterial(fY=500, E=200e3, kappa=0.05, gamma=1)
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

    sigma_ext = list_2_tensor([-100, -100, -100, 0, 0, 0])

    # Create and solve problem
    total_equilibrium = sigma_ext == mp_fesa.sigma.tensor
    prob = cp.Problem(
        objective=cp.Minimize(mp_fesa.objective),
        constraints=mp_fesa.constraint_list + [total_equilibrium],
    )
    prob.solve(verbose=False, solver="MOSEK")

    print(f"Objective value: {prob.value}")
    print(f"sigma:\n{mp_fesa.sigma.tensor.value}")
    print(f"sc:\n{mp_fesa.stress_point_dict['sc'].tensor.value}")
    print(f"sc_l:\n{mp_fesa.stress_point_dict['sc_l'].tensor.value}")
    print(f"ps_l:\n{mp_fesa.aux_variable_dict['ps_l'].value}")
    print(f"sc_h:\n{mp_fesa.stress_point_dict['sc_h'].tensor.value}")
    print(f"ps_h:\n{mp_fesa.aux_variable_dict['ps_h'].value}")
    print(f"ss:\n{mp_fesa.stress_point_dict['ss'].tensor.value}")
    for i, rs in enumerate(mp_fesa.reinforcement_stress_list):
        print(f"reinforcement {i}")
        rs.print_values()

    print([val.value for val in mp_fesa.complimentary_energy_variable_list])
