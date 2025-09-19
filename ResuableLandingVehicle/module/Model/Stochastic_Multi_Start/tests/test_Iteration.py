from .test_Initial import test_Feasible as initial_test_Feasible

from ..Multi_Start_Iterate_Model import Multi_Start_Iterate_Model

import pyomo.kernel as pmo


def test_Feasible(n_iter: int = 10, headless=True):
    model_i = initial_test_Feasible(headless=False)

    for i in range(n_iter):
        model_i = Multi_Start_Iterate_Model(
            subModels=model_i.subModels, num_non_anticipative_periods=5
        )
        solver = pmo.SolverFactory("gurobi")
        results = solver.solve(model_i, tee=False)
        if results.solver.termination_condition == pmo.TerminationCondition.error:
            print(f"Iteration {i}: Error in solver.")
            print(results)
            print("Terminating.")
            break

    if headless:
        import matplotlib

        matplotlib.use("TkAgg")
        model_i.Plot(saveFileName="MultiStart_Iteration.gif")

    return model_i
