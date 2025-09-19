from ..Step_Model import Initial_Step_Model
from ...System import SystemParameters
from ..State import State, InitializationState
from .StateTree import State_Node

import pyomo.kernel as pmo
from typing import Iterator


class Initial_Node(Initial_Step_Model):
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
        yield self
        for child in self.child_nodes:
            yield from child.iter_nodes()

    def iter_leaf_nodes(self) -> Iterator["Initial_Node"]:
        if len(self.child_nodes) == 0:
            yield self
        else:
            for child in self.child_nodes:
                yield from child.iter_leaf_nodes()

    def getIterationStates(self):
        node = State_Node(self.getIterationState())
        for child in self.child_nodes:
            node.children.append(child.getIterationStates())
        return node
