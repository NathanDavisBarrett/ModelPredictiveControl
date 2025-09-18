from .Initial_Node import Initial_Node
from .Base_Model import Base_Model
from ...System import SystemParameters

from typing import Union
import pyomo.kernel as pmo


class Initial_Model(Base_Model):
    def __init__(
        self,
        params: SystemParameters,
        max_depth: int,
        start: float,
        dt: Union[float, None],
    ):
        super().__init__(
            params,
            max_depth,
            start,
            dt,
        )

        self.root = Initial_Node(
            params=params,
            dt=self.dt,
            depth=0,
            max_depth=self.max_depth,
            prevState=self.initState,
            initializationState=None,
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

        self.objective = pmo.objective(
            expr=(self.artificial_acceleration_cost + self.mass_cost),
            sense=pmo.minimize,
        )

    def propagate_two_stage(
        self,
        stage1Length: int,
        stage2Length: int,
        numStage2Scenarios: int,
    ):
        assert stage1Length > 0, "Stage 1 length must be greater than 0"
        assert stage2Length > 0, "Stage 2 length must be greater than 0"
        assert (
            numStage2Scenarios > 0
        ), "Number of stage 2 scenarios must be greater than 0"
        self.max_depth = stage1Length + stage2Length

        self.root.max_depth = self.max_depth

        node_i = self.root
        for _ in range(1, stage1Length):
            node_i = node_i.add_child()
        for _ in range(numStage2Scenarios):
            node_j = node_i
            for _ in range(stage2Length):
                node_j = node_j.add_child()

        self.finalize()

    def propagate_linear(
        self,
        totalDepth: int,
    ):
        assert totalDepth > 0, "Total depth must be greater than 0"

        self.root.max_depth = totalDepth

        node_i = self.root
        for _ in range(1, totalDepth):
            node_i = node_i.add_child()

        self.finalize()
