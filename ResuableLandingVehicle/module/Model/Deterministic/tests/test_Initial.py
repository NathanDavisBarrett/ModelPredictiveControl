from .testing_Parameters import N_STEPS
from ..Initial_Model import Initial_Parameters, Initial_Model

import pyomo.environ as pmo
from PyomoTools.base.Solvers import WrappedSolver


def test_Initial_Construction():
    # T_guess = np.load("landing_thrust.npy")
    params = Initial_Parameters()
    model = Initial_Model(params, nSteps=N_STEPS, start=0, stop=30)
    assert model is not None
    return params, model


def test_Initial_Feasible(headless=True):
    params, model = test_Initial_Construction()

    solver = WrappedSolver(
        pmo.SolverFactory("gurobi"), interactiveInfeasibilityReport=headless
    )
    results = solver.solve(model, tee=headless)
    assert results.solver.termination_condition == pmo.TerminationCondition.optimal
    assert pmo.value(model.artificial_acceleration_norm) < 1.0

    if headless:
        import matplotlib

        matplotlib.use("TkAgg")
        model.Plot()

    return params, model.getIterationStates(), pmo.value(model.dt)
