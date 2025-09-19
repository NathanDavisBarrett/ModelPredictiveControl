from .testing_Parameters import N_STEPS
from ..Initial_Model import Initial_Model
from ..Iterate_Model import Iterate_Model

from ...Parameters import Initial_Parameters, Iterate_Parameters

import pyomo.kernel as pmo
from PyomoTools.base.Solvers import WrappedSolver
import numpy as np


def test_n_iterable_run(n_iter=10, headless=True):
    params = Initial_Parameters()
    init_model = Initial_Model(params, max_depth=N_STEPS, start=0, dt=0.5)

    stage1Len = int(np.floor(N_STEPS / 3))
    init_model.propagate_two_stage(stage1Len, N_STEPS - stage1Len, 10)

    solver = WrappedSolver(
        pmo.SolverFactory("gurobi"), interactiveInfeasibilityReport=headless
    )
    results = solver.solve(init_model, tee=False)
    assert results.solver.termination_condition == pmo.TerminationCondition.optimal

    model_i = init_model

    # import matplotlib
    # import matplotlib.pyplot as plt

    # matplotlib.use("TkAgg")
    # fig, axes = plt.subplots(1, 3)

    # for node in model_i.iter_leaf_nodes():
    #     print(node.wind_velocity)
    #     print(node.params.wind_function.random_state)
    #     node.params.wind_function.Plot(axes)
    # plt.show()

    # STOP

    artificial_acc = None
    for i in range(n_iter):
        model_i = Iterate_Model(previousIterationModel=model_i)

        if artificial_acc is not None:
            model_i.artificial_acceleration_limit = pmo.constraint(
                model_i.artificial_acceleration_cost <= artificial_acc
            )

        solver1 = pmo.SolverFactory("gurobi")
        results = solver1.solve(model_i, tee=False, options={"NumericFocus": 3})
        print(f"Iteration {i} solve status: ", results.solver.termination_condition)

        if results.solver.termination_condition == pmo.TerminationCondition.error:
            print("Error in solver.")
            print(results)
            print("Terminating.")
            break

        artificial_acc = pmo.value(model_i.artificial_acceleration_cost)
        print(f"Iteration {i} artificial acc norm: ", artificial_acc)
        # print(f"Iteration {i} solved in {results.solver.time} s")

    # from PyomoTools.kernel import InfeasibilityReport_Interactive

    # rep = InfeasibilityReport_Interactive(model_i)
    # rep.show()

    if headless:
        import matplotlib

        matplotlib.use("TkAgg")
        model_i.Plot()
