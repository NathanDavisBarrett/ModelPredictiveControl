from System import SystemParameters
from LosslessConvexificationModel import LosslessConvexificationModel
import pyomo.kernel as pmo


def test_Construction():
    params = SystemParameters()
    model = LosslessConvexificationModel(params, start=0, stop=10, nSteps=20)
    assert model is not None


def test_Feasible():
    params = SystemParameters()
    model = LosslessConvexificationModel(params, start=0, stop=100, nSteps=100)

    solver = pmo.SolverFactory("ipopt")
    results = solver.solve(model, tee=True)
    assert results.solver.termination_condition == pmo.TerminationCondition.optimal


if __name__ == "__main__":
    test_Feasible()
