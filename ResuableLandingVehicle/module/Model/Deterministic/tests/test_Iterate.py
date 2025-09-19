from .test_Initial import test_Initial_Feasible
from .testing_Parameters import N_STEPS
from ..Iterate_Model import Iterate_Parameters, Iterate_Model

import pyomo.environ as pmo
from warnings import warn


def test_Iterate_Construction():
    initParams, initStates, dt = test_Initial_Feasible(headless=False)

    params = Iterate_Parameters.from_initial_params(initParams, initStates, dt)

    model = Iterate_Model(params, nSteps=N_STEPS, start=0)
    assert model is not None
    return params, model


def test_Iterate_Feasible(headless=True):
    params, model = test_Iterate_Construction()

    # from PyomoTools.kernel import InfeasibilityReport_Interactive
    # rep = InfeasibilityReport_Interactive(model)
    # rep.show()
    # STOP

    solver = pmo.SolverFactory("gurobi")
    result = solver.solve(
        model,
        tee=headless,
    )
    assert pmo.value(model.artificial_acceleration_norm) < 1.0
    if result.solver.termination_condition != pmo.TerminationCondition.optimal:
        warn(f"Solver did not find optimal solution: {result}")

    if headless:
        import matplotlib

        matplotlib.use("TkAgg")
        model.Plot()

    return params, model.getIterationStates()
