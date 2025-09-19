from .test_Initial import test_Feasible as initial_test_Feasible

from ..Multi_Start_Iterate_Model import Multi_Start_Iterate_Model

import pyomo.kernel as pmo


def test_Construction():
    initModel = initial_test_Feasible(headless=False)

    iterateModel = Multi_Start_Iterate_Model(
        subModels=initModel.subModels, num_non_anticipative_periods=5
    )
    return iterateModel


def test_Feasible(headless=True):
    model = test_Construction()
    solver = pmo.SolverFactory("gurobi")
    results = solver.solve(model, tee=True)
    # assert results.solver.termination_condition == pmo.TerminationCondition.optimal

    if headless:
        import matplotlib

        matplotlib.use("TkAgg")
        model.Plot()

    return model
