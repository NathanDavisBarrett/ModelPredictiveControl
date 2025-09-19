"""
Initial_Node
============

This module defines the `Initial_Node` class, which extends the `Initial_Step_Model` to represent a node in the stochastic tree for the initial model.
"""

from ..Step_Model import Initial_Step_Model
from ...System import SystemParameters
from ..State import State, InitializationState
from .StateTree import State_Node

import pyomo.kernel as pmo
from typing import Iterator


class Initial_Node(Initial_Step_Model):
    """
    Initial_Node
    ------------

    Extends the `Initial_Step_Model` to represent a node in the stochastic tree for the initial model.

    Attributes:
        t (float): Time associated with the node.
        params (SystemParameters): System parameters for the node.
        dt (float): Fixed time step duration for the initial guess.
        depth (int): Depth of the node in the stochastic tree.
        max_depth (int): Maximum depth of the stochastic tree.
        probability (float): Probability associated with the node.
        child_nodes (pmo.block_list): List of child nodes.
    """

    def __init__(
        self,
        t: float,
        params: SystemParameters,
        dt: float,  # Float must be fixed for the initial guess
        depth: int,
        max_depth: int,
        prevState: State = None,
        initializationState: InitializationState = None,
    ):
        """
        Initializes the `Initial_Node` with the given parameters.

        Args:
            t (float): Time associated with the node.
            params (SystemParameters): System parameters for the node.
            dt (float): Fixed time step duration for the initial guess.
            depth (int): Depth of the node in the stochastic tree.
            max_depth (int): Maximum depth of the stochastic tree.
            prevState (State, optional): State of the system in the previous step. Defaults to None.
            initializationState (InitializationState, optional): Initial state of the system. Defaults to None.
        """

        # NOTE: DO NOT create any class attributes before calling super().__init__(). Pyomo will crash if you do.

        initialSpeed = params.initial_speed
        finalSpeed = params.final_speed
        dsdi = (finalSpeed - initialSpeed) / max_depth
        dmdi = (params.m_dry - params.m0) / max_depth

        reference_mass = params.m0 + dmdi * (depth)
        reference_speed = initialSpeed + dsdi * (depth)

        isFinal = depth == max_depth - 1

        super().__init__(
            t_est=t,
            params=params,
            dt=dt,
            reference_mass=reference_mass,
            reference_speed=reference_speed,
            prevState=prevState,
            isFinal=isFinal,
            initializationState=initializationState,
        )

        self.t = t
        self.depth = depth
        self.max_depth = max_depth

        self.probability = None

        self.child_nodes = pmo.block_list()

    def setProbability(self):
        """
        Sets the probability for the node and propagates it to child nodes.

        Notes:
            - The root node is assigned a probability of 1.0.
            - Child nodes inherit probabilities based on the number of siblings.
        """
        if self.depth == 0:
            self.probability = 1.0
        else:
            parent = (
                self.parent.parent
            )  # Initial_Node is wrapped in a block, so go up two levels
            self.probability = parent.probability / len(parent.child_nodes)

        for child in self.child_nodes:
            child.setProbability()

    def add_child(
        self,
        params: SystemParameters = None,
        initializationState: InitializationState = None,
    ) -> "Initial_Node":
        """
        Adds a child node to the current node.

        Args:
            params (SystemParameters, optional): System parameters for the child node. Defaults to None.
            initializationState (InitializationState, optional): Initial state for the child node. Defaults to None.

        Returns:
            Initial_Node: The newly created child node.

        Raises:
            ValueError: If the node is at the maximum depth.
        """
        if self.depth >= self.max_depth:
            raise ValueError("Cannot add child to node at max depth")

        child = Initial_Node(
            t=self.t + self.dt,
            params=params if params is not None else self.params,
            dt=self.dt,  # Float must be fixed for the initial guess
            depth=self.depth + 1,
            max_depth=self.max_depth,
            prevState=self.getState(),
            initializationState=initializationState,
        )
        self.child_nodes.append(child)
        return child

    def iter_nodes(self) -> Iterator["Initial_Node"]:
        """
        Iterates over all nodes in the subtree rooted at the current node.

        Yields:
            Iterator[Initial_Node]: An iterator over all nodes in the subtree.
        """
        yield self
        for child in self.child_nodes:
            yield from child.iter_nodes()

    def iter_leaf_nodes(self) -> Iterator["Initial_Node"]:
        """
        Iterates over all leaf nodes in the subtree rooted at the current node.

        Yields:
            Iterator[Initial_Node]: An iterator over all leaf nodes in the subtree.
        """
        if len(self.child_nodes) == 0:
            yield self
        else:
            for child in self.child_nodes:
                yield from child.iter_leaf_nodes()

    def iter_nodes_at_depth(self, depth: int) -> Iterator["Initial_Node"]:
        """
        Iterates over all nodes at a specific depth in the subtree rooted at the current node.

        Args:
            depth (int): Depth level to iterate over.

        Yields:
            Iterator[Initial_Node]: An iterator over nodes at the specified depth.
        """
        if self.depth == depth:
            yield self
        elif self.depth < depth:
            for child in self.child_nodes:
                yield from child.iter_nodes_at_depth(depth)

    def getIterationStates(self):
        """
        Retrieves the iteration states of the subtree rooted at the current node.

        Returns:
            State_Node: A tree structure containing iteration states.
        """
        node = State_Node(self.getIterationState())
        for child in self.child_nodes:
            node.children.append(child.getIterationStates())
        return node
