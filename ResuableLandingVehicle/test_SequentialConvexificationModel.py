from SequentialConvexificationModel import (
    SequentialConvexification_Initial_Model,
    SequentialConvexification_Initial_Parameters,
    SequentialConvexification_Iterate_Model,
    SequentialConvexification_Iterate_Parameters,
)

import pyomo.kernel as pmo
import numpy as np
from PyomoTools.base.Solvers import WrappedSolver

N_STEPS = 30


def test_Initial_Construction():
    # T_guess = np.load("landing_thrust.npy")
    params = SequentialConvexification_Initial_Parameters()
    model = SequentialConvexification_Initial_Model(
        params, nSteps=N_STEPS, start=0, stop=30
    )
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

    return params, model.getIterationStates()


def test_Iterate_Construction():
    initParams, initStates = test_Initial_Feasible(headless=False)

    params = SequentialConvexification_Iterate_Parameters.from_initial_params(
        initParams, initStates
    )

    model = SequentialConvexification_Iterate_Model(params, nSteps=N_STEPS, start=0)
    assert model is not None
    return params, model


def test_Iterate_Feasible(headless=True):
    params, model = test_Iterate_Construction()

    # from PyomoTools.kernel import InfeasibilityReport_Interactive
    # rep = InfeasibilityReport_Interactive(model)
    # rep.show()
    # STOP

    solver = pmo.SolverFactory("ipopt")
    results = solver.solve(
        model,
        tee=headless,
        symbolic_solver_labels=True,
        options={"halt_on_ampl_error": "yes"},
    )
    # assert results.solver.termination_condition == pmo.TerminationCondition.optimal
    # assert pmo.value(model.artificial_acceleration_norm) < 1.0

    model.Plot()

    return params, model.getIterationStates()


if __name__ == "__main__":
    test_Iterate_Feasible()
