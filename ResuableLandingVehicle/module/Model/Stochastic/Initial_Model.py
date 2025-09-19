"""
Initial_Model
=============

This module defines the `Initial_Model` class, which extends the `Base_Model` to provide specific implementations for the initial stochastic model.
"""

from .Initial_Node import Initial_Node
from .Base_Model import Base_Model
from ...System import SystemParameters

from typing import Union
import pyomo.kernel as pmo


class Initial_Model(Base_Model):
    """
    Initial_Model
    -------------

    Extends the `Base_Model` to provide specific implementations for the initial stochastic model.

    Attributes:
        params (SystemParameters): System parameters for the model.
        max_depth (int): Maximum depth of the stochastic tree.
        start (float): Start time of the simulation.
        dt (Union[float, pmo.variable]): Time step duration.
        root (Initial_Node): Root node of the stochastic tree.
    """

    def __init__(
        self,
        params: SystemParameters,
        max_depth: int,
        start: float,
        dt: Union[float, None],
    ):
        """
        Initializes the `Initial_Model` with the given parameters.

        Args:
            params (SystemParameters): System parameters for the model.
            max_depth (int): Maximum depth of the stochastic tree.
            start (float): Start time of the simulation.
            dt (Union[float, None]): Time step duration. If None, the time step is variable.
        """
        super().__init__(
            params,
            max_depth,
            start,
            dt,
        )

        self.root = Initial_Node(
            t=self.start,
            params=params,
            dt=self.dt,
            depth=0,
            max_depth=self.max_depth,
            prevState=self.initState,
            initializationState=None,
        )

    def finalize(self):
        """
        Finalizes the stochastic model by setting probabilities and defining the objective function.

        Notes:
            - The artificial acceleration cost departs from the original paper by using the sum of squares instead of the 2-norm. This penalizes large spikes and allows the solver to handle the model more efficiently.
        """
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

        self.objective_expr = pmo.expression(
            self.artificial_acceleration_cost + self.mass_cost
        )

        self.objective = pmo.objective(
            expr=self.objective_expr,
            sense=pmo.minimize,
        )

    def propagate_two_stage(
        self,
        stage1Length: int,
        stage2Length: int,
        numStage2Scenarios: int,
    ):
        """
        Propagates the stochastic tree in two stages.

        Args:
            stage1Length (int): Length of the first stage.
            stage2Length (int): Length of the second stage.
            numStage2Scenarios (int): Number of scenarios in the second stage.

        Notes:
            - Different weather patterns are assigned to each scenario in the second stage.
        """
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
            for j in range(stage2Length):
                node_j = node_j.add_child()
                # When breaking off into scenarios, assign different weather patterns to each one.
                if j == 0:
                    node_j.params = node_j.params.spawn(pmo.value(node_j.t_est))

        self.finalize()

    def propagate_linear(
        self,
        totalDepth: int,
    ):
        """
        Propagates the stochastic tree linearly to the specified depth.

        Args:
            totalDepth (int): Total depth of the stochastic tree.
        """
        assert totalDepth > 0, "Total depth must be greater than 0"

        self.root.max_depth = totalDepth

        node_i = self.root
        for _ in range(1, totalDepth):
            node_i = node_i.add_child()

        self.finalize()
