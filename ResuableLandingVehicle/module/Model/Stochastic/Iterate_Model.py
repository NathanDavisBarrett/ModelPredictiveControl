from .Base_Model import Base_Model
from .Iterate_Node import Iterate_Node
from .StateTree import State_Node
from ..State import IterationState
from ..Parameters import Iterate_Parameters, Initial_Parameters
from ..Step_Model import Base_Step_Model

from typing import Iterable
import pyomo.kernel as pmo


class Iterate_Model(Base_Model):
    def __init__(
        self,
        previousIterationModel: Base_Model,
    ):
        params = self.get_params_from_model(previousIterationModel)
        super().__init__(
            params,
            max_depth=previousIterationModel.max_depth,
            start=previousIterationModel.start,
            dt=None,
        )

        self.previousIterationStates = previousIterationModel.getIterationStates()

        rootParams = self.get_params_from_model(previousIterationModel.root)

        self.root = Iterate_Node(
            t_est=self.start,
            params=rootParams,
            dt=self.dt,
            depth=0,
            max_depth=self.max_depth,
            prevIterationState=self.previousIterationStates.root.state,
            prevTimeState=self.initState,
        )

        self.propagate_initialization(
            node=self.root,
            childStateNodes=self.previousIterationStates.root.children,
            prevIterationNode=previousIterationModel.root,
        )

        self.finalize()

    @staticmethod
    def get_params_from_model(model: Base_Model) -> Iterate_Parameters:
        if isinstance(model.params, Initial_Parameters):
            return Iterate_Parameters.from_initial_params(
                model.params,
                model.getIterationStates(),
                dt_est=pmo.value(model.dt),
            )
        elif isinstance(model.params, Iterate_Parameters):
            params = model.params
            params.dt_est = pmo.value(model.dt)
            return params
        else:
            raise ValueError(f"Unknown parameters type: {type(model.params)}")

    def propagate_initialization(
        self,
        node: Iterate_Node,
        childStateNodes: Iterable[State_Node],
        prevIterationNode: Base_Step_Model,
    ):
        for i in range(len(childStateNodes)):
            prevIterationNode_i = prevIterationNode.child_nodes[i]
            prevIterParams = self.get_params_from_model(prevIterationNode_i)

            node_i = Iterate_Node(
                t_est=node.t_est + pmo.value(self.dt),
                params=prevIterParams,
                dt=self.dt,
                depth=node.depth + 1,
                max_depth=self.max_depth,
                prevIterationState=childStateNodes[i].state,
                prevTimeState=node.getState(),
            )
            node.child_nodes.append(node_i)
            self.propagate_initialization(
                node_i, childStateNodes[i].children, prevIterationNode_i
            )

    def finalize(self):
        super().finalize()

        self.artificial_acceleration_cost = pmo.expression(
            sum(
                [
                    node.params.w_a
                    * node.artificial_acceleration_mag**2
                    * node.probability
                    for node in self.iter_nodes()
                ]
            )
        )  # DEPARTING FROM THE ORIGINAL PAPER HERE. Originally, the 2-norm, but the sum of squares is better to penalize large spikes and allows the gurobi solver so solve the model very quickly (since the objective is now quadratic).

        self.mass_cost = pmo.expression(
            sum(
                [
                    -node.params.w_m * node.mass * node.probability
                    for node in self.iter_leaf_nodes()
                ]
            )
        )

        self.thrust_change_cost = pmo.expression(
            sum(
                [
                    node.params.w_eta_dt * node.eta_thrust**2
                    for node in self.iter_nodes()
                ]
            )
        )  # DEPARTING FROM THE ORIGINAL PAPER HERE. Originally, the 2-norm, but the sum of squares is better to penalize large spikes.

        # TODO: Different scenarios (steps) should have different dts.
        self.eta_dt = pmo.variable(domain=pmo.NonNegativeReals, value=1.0)

        dt_change = self.dt - self.previousIterationStates.root.state.dt
        self.eta_dt_def = pmo.constraint(dt_change**2 <= self.eta_dt)

        self.objective_expr = pmo.expression(
            self.mass_cost
            + self.thrust_change_cost
            + self.eta_dt * self.params.w_eta_dt
            + self.artificial_acceleration_cost
        )

        self.objective = pmo.objective(
            expr=self.objective_expr,
            sense=pmo.minimize,
        )
