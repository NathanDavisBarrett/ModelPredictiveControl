from .testing_Parameters import N_STEPS
from ..Initial_Model import Initial_Parameters, Initial_Model
from ..Iterate_Model import Iterate_Parameters, Iterate_Model

import pyomo.kernel as pmo
from PyomoTools.base.Solvers import WrappedSolver


def test_n_iterable_run(n_iter=10, headless=True):
    params = Initial_Parameters()
    init_model = Initial_Model(params, nSteps=N_STEPS, start=0, stop=30)
    solver = WrappedSolver(
        pmo.SolverFactory("gurobi"), interactiveInfeasibilityReport=headless
    )
    results = solver.solve(init_model, tee=headless)
    assert results.solver.termination_condition == pmo.TerminationCondition.optimal

    model_i = init_model
    artificial_acc = None
    for i in range(n_iter):
        params_i = Iterate_Parameters.from_initial_params(
            params, model_i.getIterationStates(), dt_est=pmo.value(model_i.dt)
        )
        model_i = Iterate_Model(params_i, nSteps=N_STEPS, start=0)

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

    # from PyomoTools.kernel import InfeasibilityReport_Interactive

    # rep = InfeasibilityReport_Interactive(model_i)
    # rep.show()

    if headless:
        import matplotlib

        matplotlib.use("TkAgg")
        model_i.Plot()
