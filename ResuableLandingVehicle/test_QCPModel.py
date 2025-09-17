from System import SystemParameters
from QCPModel import QCPModel
import pyomo.kernel as pmo


def test_Construction():
    params = SystemParameters()
    model = QCPModel(params, start=0, stop=10, nSteps=20)
    assert model is not None


def test_Feasible():
    params = SystemParameters()
    model = QCPModel(params, start=0, stop=10, nSteps=20)

    solver = pmo.SolverFactory("gurobi")
    results = solver.solve(model, tee=True)
    assert results.solver.termination_condition == pmo.TerminationCondition.optimal


if __name__ == "__main__":
    test_Feasible()
