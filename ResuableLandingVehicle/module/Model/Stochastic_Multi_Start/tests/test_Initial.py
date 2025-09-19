from ..Multi_Start_Initial_Model import Multi_Start_Initial_Model
from ...Stochastic import Initial_Model
from ...Parameters import Initial_Parameters

from .testing_Parameters import N_STEPS

import pyomo.kernel as pmo


def construct_base_linear_models(numModels: int = 2, spawn_time: float = 10):
    params = [Initial_Parameters()]
    for i in range(1, numModels):
        params.append(params[0].spawn(spawn_time))

    subModels = [
        Initial_Model(params[i], max_depth=N_STEPS, start=0, dt=0.5)
        for i in range(numModels)
    ]

    for model in subModels:
        model.propagate_linear(N_STEPS)

    return subModels


def construct_base_two_stage_models(numModels: int = 5, spawn_time: float = 10):
    params = [Initial_Parameters()]
    for i in range(1, numModels):
        params.append(params[0].spawn(spawn_time))

    subModels = [
        Initial_Model(params[i], max_depth=N_STEPS, start=0, dt=0.5)
        for i in range(numModels)
    ]

    for model in subModels:
        stage1Length = N_STEPS // 2
        stage2Length = N_STEPS - stage1Length
        numChildScenarios = 3
        model.propagate_two_stage(stage1Length, stage2Length, numChildScenarios)

    return subModels


def test_construction():
    subModels = construct_base_two_stage_models()
    model = Multi_Start_Initial_Model(
        subModels, num_non_anticipative_periods=N_STEPS // 3
    )
    return model


def test_Feasible(headless=True):
    model = test_construction()
    solver = pmo.SolverFactory("gurobi")
    results = solver.solve(model, tee=True)
    assert results.solver.termination_condition == pmo.TerminationCondition.optimal

    if headless:
        import matplotlib

        matplotlib.use("TkAgg")
        model.Plot()  # saveFileName="MultiStart_Initial.gif")

    return model
