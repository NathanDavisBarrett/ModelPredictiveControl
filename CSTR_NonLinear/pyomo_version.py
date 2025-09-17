import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Integral
from dataclasses import dataclass
from System import System

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


@dataclass
class Parameters:
    T_max = 400  # C

    T_jacket_bounds = (250, 350)  # K
    T_jacket_0 = 305  # K
    T_jacket_ramp_rate = 5  # K / sec

    Vdot = 100  # m^3 / sec
    V = 100  # m^3

    rho = 1000  # kg/m^3
    Cp = 0.239  # J / kg K

    dH_rxn = 5e4  # J / mol

    R = 8.314  # J / mol K
    E = 8750 * R  # J / mol

    k0 = 7.2e10  # 1 / sec (pre-exponential factor)
    UA = 5e4  # J / sec m^2 K (overall heat transfer coefficient * area)

    C_in = 1  # mol / m^3 (inlet concentration)
    T_in = 350  # C (inlet temperature)

    C_0 = 0.9  # mol / m^3 (initial control-volume concentration)
    T_0 = 305  # K  (initial control-volume temperature)

    def setPoint(self, t):
        return (t / 20) ** 3 - (t / 19) ** 2 + 0.9


def solve(params, start, stop):
    model = pyo.ConcreteModel()

    model.TIME = ContinuousSet(bounds=(start, stop))

    ## VARIABLES
    model.C = pyo.Var(
        model.TIME, bounds=(0, 1), initialize=1
    )  # Control volume concentration
    model.T = pyo.Var(
        model.TIME, bounds=(0, 2000), initialize=params.T_0
    )  # Control volume temperature (K)
    model.k = pyo.Var(
        model.TIME,
        initialize=params.k0 * pyo.exp(-params.E / (params.R * (params.T_0))),
    )  # Reaction rate constant
    model.rate = pyo.Var(
        model.TIME,
        domain=pyo.NonNegativeReals,
        initialize=params.k0
        * pyo.exp(-params.E / (params.R * (params.T_0)))
        * params.C_0,
    )  # Reaction rate
    model.T_jacket = pyo.Var(
        model.TIME, bounds=params.T_jacket_bounds, initialize=params.T_jacket_0
    )  # Jacket temperature (K)

    model.dCdt = DerivativeVar(model.C, wrt=model.TIME)
    model.dTdt = DerivativeVar(model.T, wrt=model.TIME)
    model.dTJdt = DerivativeVar(model.T_jacket, wrt=model.TIME)

    model.Err = pyo.Var(model.TIME, initialize=0)
    model.TotalErr = Integral(model.TIME, wrt=model.TIME, rule=lambda m, t: m.Err[t])

    ## INITIAL CONDITIONS
    model.C_0 = pyo.Constraint(expr=model.C[start] == params.C_0)
    model.T_0 = pyo.Constraint(expr=model.T[start] == params.T_0)
    model.T_jacket_0 = pyo.Constraint(expr=model.T_jacket[start] == params.T_jacket_0)

    ## CONSTRAINTS
    model.k_def = pyo.Constraint(
        model.TIME,
        rule=lambda m, t: m.k[t]
        == params.k0 * pyo.exp(-params.E / (params.R * (m.T[t]))),
    )

    model.rate_def = pyo.Constraint(
        model.TIME, rule=lambda m, t: m.rate[t] == m.k[t] * m.C[t]
    )

    def A_Balance(m, t):
        return (
            params.V * m.dCdt[t]
            == params.Vdot * (params.C_in - m.C[t]) - params.V * m.rate[t]
        )

    model.A_Balance = pyo.Constraint(model.TIME, rule=A_Balance)

    def Energy_Balance(m, t):
        return params.V * params.rho * params.Cp * m.dTdt[
            t
        ] == params.Vdot * params.rho * params.Cp * (
            params.T_in - m.T[t]
        ) + params.dH_rxn * params.V * m.rate[
            t
        ] + params.UA * (
            m.T_jacket[t] - m.T[t]
        )

    model.Energy_Balance = pyo.Constraint(model.TIME, rule=Energy_Balance)

    def Jacket_Ramping_Limit(m, t, uplow):
        if uplow == 0:
            return m.dTJdt[t] <= params.T_jacket_ramp_rate  # K / sec
        else:
            return m.dTJdt[t] >= -params.T_jacket_ramp_rate  # K / sec

    model.Jacket_Ramping_Limit_Up = pyo.Constraint(
        model.TIME, [0, 1], rule=Jacket_Ramping_Limit
    )

    model.Err_def = pyo.Constraint(
        model.TIME, rule=lambda m, t: m.Err[t] == (m.C[t] - params.setPoint(t)) ** 2
    )

    model.Obj = pyo.Objective(
        expr=model.TotalErr,
        sense=pyo.minimize,
    )

    discretizer = pyo.TransformationFactory("dae.finite_difference")
    discretizer.apply_to(model, nfe=200, wrt=model.TIME, scheme="BACKWARD")

    solver = pyo.SolverFactory("ipopt")
    results = solver.solve(model)
    # assert (
    #     results.solver.termination_condition == pyo.TerminationCondition.optimal
    # ), "Error: Solver did not converge to optimal solution: Termination condition = {}".format(
    #     results.solver.termination_condition
    # )

    return model


params = Parameters()
model = solve(params, 0, 2)

t_sim = 20  # seconds
t_re_optimize = 1  # seconds
t_horizon = 2  # seconds

num_re_optimizations = int(t_sim / t_re_optimize)

system = System()


system.C = params.C_0
system.T = params.T_0

fig, (JacketAx, CAx, TAx) = plt.subplots(3, 1)

for p in [JacketAx, CAx, TAx]:
    p.set_xlabel("Time (s)")
    p.set_xlim(0, t_sim)

JacketAx.set_ylabel("Jacket\nTemperature\n(K)")
CAx.set_ylabel("Concentration\n(mol/m^3)")
TAx.set_ylabel("Temperature\n(K)")

JacketAx.set_ylim(params.T_jacket_bounds)
CAx.set_ylim(0, 1)
TAx.set_ylim(200, 400)

JacketAx.plot([0, 0], [0, 0], "--", label="Optimized Trajectory", color="orange")
JacketAx.plot([0, 0], [0, 0], "-", label="Actual Trajectory", color="black")

CAx.plot([0, 0], [0, 0], "--", label="Optimized Trajectory", color="orange")
CAx.plot([0, 0], [0, 0], "-", label="Set Point", color="green")
CAx.plot([0, 0], [0, 0], "-", label="Actual Trajectory", color="black")

TAx.plot([0, 0], [0, 0], "--", label="Optimized Trajectory", color="orange")
TAx.plot([0, 0], [0, 0], "-", label="Actual Trajectory", color="black")

JacketAx.legend(loc="lower left")
CAx.legend(loc="lower left")
TAx.legend(loc="lower left")

jacketPlot = None
CPlot = None
TPlot = None

for i in tqdm(range(num_re_optimizations)):
    start = i * t_re_optimize
    model = solve(params, start, start + t_horizon)
    n = len(model.TIME)
    TIME = [t for t in model.TIME]
    T_jacket_opt = [pyo.value(model.T_jacket[t]) for t in model.TIME]
    C_opt = [pyo.value(model.C[t]) for t in model.TIME]
    T_opt = [pyo.value(model.T[t]) for t in model.TIME]
    C_sp = [params.setPoint(t) for t in model.TIME]

    for p in [jacketPlot, CPlot, TPlot]:
        if p is not None:
            p.remove()

    (jacketPlot,) = JacketAx.plot(TIME, T_jacket_opt, "--", color="orange")
    (CPlot,) = CAx.plot(TIME, C_opt, "--", color="orange")
    (TPlot,) = TAx.plot(TIME, T_opt, "--", color="orange")

    for ii in range(len(T_jacket_opt) - 1):
        t = TIME[ii]
        if t >= start + t_re_optimize:
            break

        dt = TIME[ii + 1] - TIME[ii]

        CAx.plot([t, t + dt], [C_sp[ii], C_sp[ii + 1]], "-", color="green")

        oldJacket = system.T_jacket
        oldC = system.C
        oldT = system.T

        system.step(T_jacket_opt[ii], dt)
        if ii % 10 == 0:
            JacketAx.plot([t, t + dt], [oldJacket, system.T_jacket], "-", color="black")
            CAx.plot([t, t + dt], [oldC, system.C], "-", color="black")
            TAx.plot([t, t + dt], [oldT, system.T], "-", color="black")

            fig.tight_layout()
            fig.savefig(f"CSTR_NonLinear/Images/CSTR_MPC_{i:05d}_{ii:05d}.png")

    params.C_0 = system.C
    params.T_0 = system.T
    params.T_jacket_0 = system.T_jacket
