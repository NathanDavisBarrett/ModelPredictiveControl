"""
Iterate_Node
============

This module defines the `Iterate_Node` class, which extends the `Iterate_Step_Model` to represent a node in the stochastic tree for iterative models.
"""

from ..Step_Model import Iterate_Step_Model
from ..Parameters import Iterate_Parameters
from ..State import State, IterationState
from .StateTree import State_Node

import pyomo.kernel as pmo
from typing import Iterator


class Iterate_Node(Iterate_Step_Model):
    """
    Iterate_Node
    ------------

    Extends the `Iterate_Step_Model` to represent a node in the stochastic tree for iterative models.

    Attributes:
        depth (int): Depth of the node in the stochastic tree.
        max_depth (int): Maximum depth of the stochastic tree.
        probability (float): Probability associated with the node.
        child_nodes (pmo.block_list): List of child nodes.
    """

    def __init__(
        self,
        t_est: float,  # Estimated time at this node
        params: Iterate_Parameters,
        dt: pmo.variable,
        depth: int,
        max_depth: int,
        prevIterationState: IterationState,
        prevTimeState: State,
    ):
        """
        Initializes the `Iterate_Node` with the given parameters.

        Args:
            t_est (float): Estimated time at this node.
            params (Iterate_Parameters): Parameters for the node.
            dt (pmo.variable): Time step duration.
            depth (int): Depth of the node in the stochastic tree.
            max_depth (int): Maximum depth of the stochastic tree.
            prevIterationState (IterationState): State from the previous iteration.
            prevTimeState (State): State from the previous time step.
        """

        # NOTE: DO NOT create any class attributes before calling super().__init__(). Pyomo will crash if you do.

        isFinal = depth == max_depth - 1

        super().__init__(
            t_est=t_est,
            params=params,
            dt=dt,
            prevTimeState=prevTimeState,
            prevIterationState=prevIterationState,
            isFinal=isFinal,
        )

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
        prevIterationState: IterationState,
        params: Iterate_Parameters = None,
    ) -> "Iterate_Node":
        """
        Adds a child node to the current node.

        Args:
            prevIterationState (IterationState): State from the previous iteration.
            params (Iterate_Parameters, optional): Parameters for the child node. Defaults to None.

        Returns:
            Iterate_Node: The newly created child node.

        Raises:
            ValueError: If the node is at the maximum depth.
        """
        if self.depth >= self.max_depth:
            raise ValueError("Cannot add child to node at max depth")

        child = Iterate_Node(
            params=params if params is not None else self.params,
            dt=self.dt,  # Float must be fixed for the initial guess
            depth=self.depth + 1,
            max_depth=self.max_depth,
            prevTimeState=self.getState(),
            prevIterationState=prevIterationState,
        )
        self.child_nodes.append(child)
        return child

    def iter_nodes(self) -> Iterator["Iterate_Node"]:
        """
        Iterates over all nodes in the subtree rooted at the current node.

        Yields:
            Iterator[Iterate_Node]: An iterator over all nodes in the subtree.
        """
        yield self
        for child in self.child_nodes:
            yield from child.iter_nodes()

    def iter_leaf_nodes(self) -> Iterator["Iterate_Node"]:
        """
        Iterates over all leaf nodes in the subtree rooted at the current node.

        Yields:
            Iterator[Iterate_Node]: An iterator over all leaf nodes in the subtree.
        """
        if len(self.child_nodes) == 0:
            yield self
        else:
            for child in self.child_nodes:
                yield from child.iter_leaf_nodes()

    def iter_nodes_at_depth(self, depth: int) -> Iterator["Iterate_Node"]:
        """
        Iterates over all nodes at a specific depth in the subtree rooted at the current node.

        Args:
            depth (int): Depth level to iterate over.

        Yields:
            Iterator[Iterate_Node]: An iterator over nodes at the specified depth.
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
