"""
StateTree
=========

This module defines the `State_Node` and `StateTree` classes, which represent nodes and trees for managing states in the stochastic optimization process.
"""

from ..State import IterationState, State, InitializationState

from typing import Union


class State_Node:
    """
    State_Node
    ----------

    Represents a node in the state tree.

    Attributes:
        state (Union[IterationState, InitializationState, State]): The state associated with the node.
        children (list): List of child nodes.
    """

    def __init__(self, state: Union[IterationState, InitializationState, State]):
        """
        Initializes the `State_Node` with the given state.

        Args:
            state (Union[IterationState, InitializationState, State]): The state associated with the node.
        """
        self.state = state
        self.children = []


class StateTree:
    """
    StateTree
    ---------

    Represents a tree structure for managing states in the stochastic optimization process.

    Attributes:
        root (State_Node): The root node of the state tree.
    """

    def __init__(self, root: State_Node):
        """
        Initializes the `StateTree` with the given root node.

        Args:
            root (State_Node): The root node of the state tree.
        """
        self.root = root
