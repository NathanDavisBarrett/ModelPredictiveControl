from System import SystemParameters

import pyomo.environ as pyo
from pyomo.dae import DerivativeVar, ContinuousSet
import numpy as np

X = 0
Y = 1
Z = 2


def FullModel(params: SystemParameters, start: float, stop: float, nSteps: int):
    model = pyo.ConcreteModel()
    model.TIME = ContinuousSet(bounds=(start, stop))
    model.DIM = pyo.Set(initialize=[X, Y, Z])  # 3D vectors

    # State variables
    model.X = pyo.Var(model.TIME, model.DIM)  # Position (m)
    model.X_MAG = pyo.Var(model.TIME)  # Position magnitude (m)
    model.V = DerivativeVar(model.X, wrt=model.TIME)  # Velocity (m/s)
    model.A = DerivativeVar(model.V, wrt=model.TIME)  # Acceleration (m/s^2)

    model.T = pyo.Var(model.TIME, model.DIM)  # Thrust vector (N)
    model.T_MAG = pyo.Var(model.TIME)  # Thrust magnitude (N)
    model.dT_MAG_dt = DerivativeVar(
        model.T_MAG, wrt=model.TIME
    )  # Thrust rate of change (N/s)
    model.D = pyo.Var(model.TIME, model.DIM)  # Drag force (N)

    model.M = pyo.Var(model.TIME)  # Mass (kg)
    model.dMdt = DerivativeVar(model.M, wrt=model.TIME)  # Mass depletion rate (kg/s)

    # Initial conditions
    model.X_init = pyo.Constraint(
        model.DIM, rule=lambda m, d: m.X[start, d] == params.x0[d]
    )
    model.V_init = pyo.Constraint(
        model.DIM, rule=lambda m, d: m.V[start, d] == params.v0[d]
    )
    # model.m_init = pyo.Constraint(rule=lambda m: m.m[start] == params.m0)

    # Terminal conditions
    model.X_final = pyo.Constraint(model.DIM, rule=lambda m, d: m.X[stop, d] == 0)
    model.V_final = pyo.Constraint(model.DIM, rule=lambda m, d: m.V[stop, d] == 0)
    model.normal_landing = pyo.Constraint(
        expr=params.math.dot([model.T[stop, d] for d in model.DIM], params.nf)
        == model.T_MAG[stop] * params.math.norm(params.nf)
    )

    # Dynamics constraints
    def X_mag_rule(m, t):
        return m.X_MAG[t] * m.X_MAG[t] == sum(m.X[t, d] * m.X[t, d] for d in m.DIM)

    model.X_magnitude = pyo.Constraint(model.TIME, rule=X_mag_rule)

    def D_rule(m, t, d):
        v = [m.V[t, X], m.V[t, Y], m.V[t, Z]]
        v_mag = params.math.norm(v)
        factor = -0.5 * params.rho * params.Cd * params.Sd * v_mag
        return m.D[t, d] == factor * m.V[t, d]

    model.D_def = pyo.Constraint(model.TIME, model.DIM, rule=D_rule)

    def NewtonsSecondLaw_rule(m, t, d):
        return m.M[t] * m.A[t, d] == m.T[t, d] + m.D[t, d] + params.g[d] * m.M[t]

    model.NewtonsSecondLaw = pyo.Constraint(
        model.TIME, model.DIM, rule=NewtonsSecondLaw_rule
    )

    def ThrustMagnitude_rule(m, t):
        return m.T_MAG[t] * m.T_MAG[t] == sum(m.T[t, d] * m.T[t, d] for d in m.DIM)

    model.ThrustMagnitude = pyo.Constraint(model.TIME, rule=ThrustMagnitude_rule)

    def MassDepletion_rule(m, t):
        IspG0 = params.I_sp * params.g_mag
        dmdt = model.dMdt[t]

        return dmdt * IspG0 == -m.T_MAG[t] - params.P * params.A_nozzle

    model.MassDepletion = pyo.Constraint(model.TIME, rule=MassDepletion_rule)

    def DryMass_rule(m, t):
        return m.M[t] >= params.m_dry

    model.DryMass = pyo.Constraint(model.TIME, rule=DryMass_rule)

    def GlideSlope_rule(m, t):
        return m.X_MAG[t] * np.cos(params.max_glide_slope) <= sum(
            m.X[t, d] * params.e_u[d] for d in m.DIM
        )

    model.GlideSlope = pyo.Constraint(model.TIME, rule=GlideSlope_rule)

    def ThrustLowerLimits_rule(m, t):
        return m.T_MAG[t] >= params.T_min

    model.ThrustLowerLimits = pyo.Constraint(model.TIME, rule=ThrustLowerLimits_rule)

    def ThrustUpperLimits_rule(m, t):
        return m.T_MAG[t] <= params.T_max

    model.ThrustUpperLimits = pyo.Constraint(model.TIME, rule=ThrustUpperLimits_rule)

    def TiltAngle_rule(m, t):
        T = [m.T[t, d] for d in m.DIM]
        return params.math.dot(params.e_u, T) >= np.cos(params.max_tilt) * m.T_MAG[t]

    model.TiltAngle = pyo.Constraint(model.TIME, rule=TiltAngle_rule)

    def ThrustUpperRampRate_rule(m, t):
        return m.dT_MAG_dt[t] <= params.dTdt_max

    model.ThrustUpperRampRate = pyo.Constraint(
        model.TIME, rule=ThrustUpperRampRate_rule
    )

    def ThrustLowerRampRate_rule(m, t):
        return m.dT_MAG_dt[t] >= params.dTdt_min

    model.ThrustLowerRampRate = pyo.Constraint(
        model.TIME, rule=ThrustLowerRampRate_rule
    )

    model.obj = pyo.Objective(expr=model.M[start], sense=pyo.minimize)

    # Discretize the model
    discretizer = pyo.TransformationFactory("dae.finite_difference")
    discretizer.apply_to(model, nfe=nSteps, wrt=model.TIME, scheme="BACKWARD")

    # Initialize variables
    for t in model.TIME:
        for d in model.DIM:
            model.X[t, d].value = params.x0[d]
            model.V[t, d].value = params.v0[d]
        model.X_MAG[t].value = params.math.norm(params.x0)
        model.M[t].value = params.m0

        D = params.ComputeDragForce(params.v0)
        T = [params.g[d] * params.m0 - D[d] for d in model.DIM]
        T_Mag = params.math.norm(T)

        for d in model.DIM:
            model.T[t, d].value = T[d]
        model.T_MAG[t].value = T_Mag
        model.dT_MAG_dt[t].value = 0.0

        model.D[t, X].value = D[0]
        model.D[t, Y].value = D[0]
        model.D[t, Z].value = D[0]

    return model
