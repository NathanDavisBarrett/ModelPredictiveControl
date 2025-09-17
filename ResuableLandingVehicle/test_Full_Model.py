from System import SystemParameters
from Full_Model import FullModel
from Plot_Full_Model import PlotFullModel

import pyomo.environ as pyo


def test_Construction():
    params = SystemParameters()
    model = FullModel(params, start=0, stop=10, nSteps=20)
    assert model is not None


def test_Feasible():
    params = SystemParameters()
    model = FullModel(params, start=0, stop=10, nSteps=20)

    solver = pyo.SolverFactory("ipopt")
    results = solver.solve(model, tee=True)
    # assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    PlotFullModel(model, params)


if __name__ == "__main__":
    test_Feasible()
