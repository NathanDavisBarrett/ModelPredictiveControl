from SequentialConvexificationModel import (
    SequentialConvexification_Initial_Model,
    SequentialConvexification_Initial_Parameters,
    SequentialConvexification_Iterate_Model,
    SequentialConvexification_Iterate_Parameters,
)

import pyomo.kernel as pmo
import numpy as np
from PyomoTools.base.Solvers import WrappedSolver

N_STEPS = 50


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

    if headless:
        import matplotlib

        matplotlib.use("TkAgg")
        model.Plot()

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

    solver = pmo.SolverFactory("gurobi")
    results = solver.solve(
        model,
        tee=headless,
    )
    # assert results.solver.termination_condition == pmo.TerminationCondition.optimal
    assert pmo.value(model.artificial_acceleration_norm) < 1.0

    if headless:
        import matplotlib

        matplotlib.use("TkAgg")
        model.Plot()

    return params, model.getIterationStates()


def test_n_iterable_run(n_iter=3, headless=True):
    params = SequentialConvexification_Initial_Parameters()
    init_model = SequentialConvexification_Initial_Model(
        params, nSteps=N_STEPS, start=0, stop=30
    )
    solver = WrappedSolver(
        pmo.SolverFactory("gurobi"), interactiveInfeasibilityReport=headless
    )
    results = solver.solve(init_model, tee=headless)
    assert results.solver.termination_condition == pmo.TerminationCondition.optimal

    model_i = init_model
    artificial_acc = None
    for i in range(n_iter):
        params_i = SequentialConvexification_Iterate_Parameters.from_initial_params(
            params, model_i.getIterationStates()
        )
        model_i = SequentialConvexification_Iterate_Model(
            params_i, nSteps=N_STEPS, start=0
        )

        if artificial_acc is not None:
            model_i.artificial_acceleration_limit = pmo.constraint(
                model_i.artificial_acceleration_norm <= artificial_acc
            )

        solver1 = pmo.SolverFactory("gurobi")
        results = solver1.solve(model_i, tee=False, options={"NumericFocus": 3})
        print(f"Iteration {i} solve status: ", results.solver.termination_condition)

        if results.solver.termination_condition == pmo.TerminationCondition.error:
            print("Error in solver.")
            print(results)
            print("Terminating.")
            break

        artificial_acc = pmo.value(model_i.artificial_acceleration_norm)
        print(f"Iteration {i} artificial acc norm: ", artificial_acc)
        # print(f"Iteration {i} solved in {results.solver.time} s")

    from PyomoTools.kernel import InfeasibilityReport_Interactive

    # rep = InfeasibilityReport_Interactive(model_i)
    # rep.show()

    if headless:
        import matplotlib

        matplotlib.use("TkAgg")
        model_i.Plot()


if __name__ == "__main__":
    test_n_iterable_run(n_iter=10, headless=True)
