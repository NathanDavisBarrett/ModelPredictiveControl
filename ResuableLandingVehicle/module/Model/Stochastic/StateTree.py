from ..State import IterationState, State, InitializationState

from typing import Union


class State_Node:
    def __init__(self, state: Union[IterationState, InitializationState, State]):
        self.state = state
        self.children = []


class StateTree:
    def __init__(self, root: State_Node):
        self.root = root
