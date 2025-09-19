"""
Base_Model
==========

This module defines the `Base_Model` class, which serves as an abstract base class for stochastic models.
It provides methods for iterating over nodes, setting probabilities, and plotting trajectories.
"""

from ...System import SystemParameters
from ..State import State  # Removed unused IterationState import
from .Initial_Node import Initial_Node
from .Iterate_Node import Iterate_Node
from .StateTree import StateTree

import pyomo.kernel as pmo
from typing import Union, Iterable, Iterator
import numpy as np
from abc import ABC

Node = Union[Initial_Node, Iterate_Node]


class Base_Model(pmo.block, ABC):
    """
    Base_Model
    ----------

    Abstract base class for stochastic models.

    Attributes:
        params (SystemParameters): System parameters for the model.
        max_depth (int): Maximum depth of the stochastic tree.
        start (float): Start time of the simulation.
        dt (Union[float, pmo.variable]): Time step duration.
        variable_dt (bool): Indicates whether the time step is variable.
        stop (float): Stop time of the simulation.
        initState (State): Initial state of the system.
    """

    def __init__(
        self,
        params: SystemParameters,
        max_depth: int,
        start: float,
        dt: Union[float, None],
    ):
        """
        Initializes the `Base_Model` with the given parameters.

        Args:
            params (SystemParameters): System parameters for the model.
            max_depth (int): Maximum depth of the stochastic tree.
            start (float): Start time of the simulation.
            dt (Union[float, None]): Time step duration. If None, the time step is variable.
        """
        super().__init__()
        self.params = params
        self.start = start
        self.max_depth = max_depth

        if dt is None:
            dt_est = params.dt_est if hasattr(params, "dt_est") else 1.0
            self.dt = pmo.variable(domain=pmo.NonNegativeReals, value=dt_est)
            self.variable_dt = True
        else:
            self.dt = dt
            self.variable_dt = False

        self.stop = self.start + self.max_depth * self.dt

        self.initState = State(
            mass=params.m0,
            position=params.x0,
            velocity=params.v0,
            acceleration=[0.0, 0.0, 0.0],
            thrust=params.T0,
            gamma=np.linalg.norm(params.T0),
        )

        # Child class must define root.

    def iter_nodes(self) -> Iterator[Node]:
        """
        Iterates over all nodes in the stochastic tree.

        Yields:
            Iterator[Node]: An iterator over all nodes.
        """
        yield from self.root.iter_nodes()

    def iter_leaf_nodes(self) -> Iterator[Node]:
        """
        Iterates over all leaf nodes in the stochastic tree.

        Yields:
            Iterator[Node]: An iterator over all leaf nodes.
        """
        yield from self.root.iter_leaf_nodes()

    def iter_nodes_at_depth(self, depth: int) -> Iterator[Node]:
        """
        Iterates over all nodes at a specific depth in the stochastic tree.

        Args:
            depth (int): Depth level to iterate over.

        Yields:
            Iterator[Node]: An iterator over nodes at the specified depth.
        """
        yield from self.root.iter_nodes_at_depth(depth)

    def iter_unique_trajectories(self) -> Iterator[Iterable[Node]]:
        """
        Iterates over all unique trajectories in the stochastic tree.

        Yields:
            Iterator[Iterable[Node]]: An iterator over unique trajectories.
        """
        # Use memoization to avoid recomputing shared ancestry
        memo = {}

        def get_lineage(node):
            if node is None:
                return []
            if node in memo:
                return memo[node]
            parent = getattr(node, "parent", None)
            if isinstance(parent, pmo.block_list):
                parent = parent.parent
            elif isinstance(parent, Base_Model):
                return [node]
            lineage = get_lineage(parent) + [node]
            memo[node] = lineage
            return lineage

        for leaf in self.iter_leaf_nodes():
            yield get_lineage(leaf)

    def setProbabilities(self):
        """
        Sets the probabilities for all nodes in the stochastic tree.
        """
        self.root.setProbability()

    def finalize(self):
        """
        Finalizes the stochastic model by setting probabilities.
        """
        self.setProbabilities()

    def getIterationStates(self) -> StateTree:
        """
        Retrieves the iteration states of the stochastic model.

        Returns:
            StateTree: A tree structure containing iteration states.
        """
        return StateTree(self.root.getIterationStates())

    def Plot(self, axDict: dict = None, withLabels: bool = True):
        """
        Plots the trajectories of the stochastic model, including position, velocity, acceleration, mass, and thrust over time.

        Args:
            axDict (dict, optional): Dictionary of axes for plotting. Defaults to None.
            withLabels (bool, optional): Whether to include labels in the plot. Defaults to True.
        """
        # import matplotlib

        # matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        dt = pmo.value(self.dt)
        times = [self.start + i * dt for i in range(self.max_depth)]

        if axDict is None:
            show = True
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(3, 2)
            posAx = fig.add_subplot(gs[0, 0])
            velAx = fig.add_subplot(gs[0, 1])
            accAx = fig.add_subplot(gs[1, 0])
            massAx = fig.add_subplot(gs[1, 1])
            thrustAx = fig.add_subplot(gs[2, 0])

            threeDimAx = fig.add_subplot(gs[2, 1], projection="3d")
        else:
            show = False
            posAx = axDict["posAx"]
            velAx = axDict["velAx"]
            accAx = axDict["accAx"]
            massAx = axDict["massAx"]
            thrustAx = axDict["thrustAx"]
            threeDimAx = axDict["threeDimAx"]

        colors = {
            "X": "blue",
            "Y": "orange",
            "Z": "green",
            "Mass": "purple",
            "Thrust": "red",
            "3D": "red",
        }

        lineKwargs = {"linewidth": 1, "alpha": 0.7}

        xmin, xmax = float("inf"), float("-inf")
        ymin, ymax = float("inf"), float("-inf")

        trajNum = -1
        for steps in self.iter_unique_trajectories():
            trajNum += 1
            positions = np.array(
                [[pmo.value(step.position[i]) for step in steps] for i in range(3)]
            )
            velocities = np.array(
                [[pmo.value(step.velocity[i]) for step in steps] for i in range(3)]
            )
            accelerations = np.array(
                [[pmo.value(step.acceleration[i]) for step in steps] for i in range(3)]
            )
            masses = np.array([pmo.value(step.mass) for step in steps])

            kwargs = {
                "X": lineKwargs.copy(),
                "Y": lineKwargs.copy(),
                "Z": lineKwargs.copy(),
                "Mass": lineKwargs.copy(),
                "Thrust": lineKwargs.copy(),
                "3D": lineKwargs.copy(),
            }
            if trajNum == 0 and withLabels:
                kwargs["X"]["label"] = "X"
                kwargs["Y"]["label"] = "Y"
                kwargs["Z"]["label"] = "Z"
                kwargs["Mass"]["label"] = "Mass"
                kwargs["Thrust"]["label"] = "Thrust"
                kwargs["3D"]["label"] = "Trajectory"

            for k in colors:
                kwargs[k]["color"] = colors[k]

            posAx.plot(times, positions[0, :], **(kwargs["X"]))
            posAx.plot(times, positions[1, :], **(kwargs["Y"]))
            posAx.plot(times, positions[2, :], **(kwargs["Z"]))

            velAx.plot(times, velocities[0, :], **(kwargs["X"]))
            velAx.plot(times, velocities[1, :], **(kwargs["Y"]))
            velAx.plot(times, velocities[2, :], **(kwargs["Z"]))

            accAx.plot(times, accelerations[0, :], **(kwargs["X"]))
            accAx.plot(times, accelerations[1, :], **(kwargs["Y"]))
            accAx.plot(times, accelerations[2, :], **(kwargs["Z"]))

            massAx.plot(times, masses, **(kwargs["Mass"]))

            thrusts = np.array(
                [[pmo.value(step.thrust[i]) for step in steps] for i in range(3)]
            )
            thrustAx.plot(times, thrusts[0, :], **(kwargs["X"]))
            thrustAx.plot(times, thrusts[1, :], **(kwargs["Y"]))
            thrustAx.plot(times, thrusts[2, :], **(kwargs["Z"]))

            threeDimAx.plot(
                positions[0, :], positions[1, :], positions[2, :], **(kwargs["3D"])
            )

            xmin = min(min(positions[0, :]), xmin)
            xmax = max(max(positions[0, :]), xmax)
            ymin = min(min(positions[1, :]), ymin)
            ymax = max(max(positions[1, :]), ymax)

        posAx.set_title("Position vs Time")
        posAx.set_xlabel("Time (s)")
        posAx.set_ylabel("Position (km)")
        posAx.legend()
        posAx.grid()

        velAx.set_title("Velocity vs Time")
        velAx.set_xlabel("Time (s)")
        velAx.set_ylabel("Velocity (km/s)")
        velAx.legend()
        velAx.grid()

        accAx.set_title("Acceleration vs Time")
        accAx.set_xlabel("Time (s)")
        accAx.set_ylabel("Acceleration (km/s²)")
        accAx.legend()
        accAx.grid()

        label = {"label": "Dry Mass"} if withLabels else {}
        massAx.axhline(self.params.m_dry, color="red", linestyle="--", **label)
        massAx.set_title("Mass vs Time")
        massAx.set_xlabel("Time (s)")
        massAx.set_ylabel("Mass (Mg)")
        massAx.legend()
        massAx.grid()

        thrustAx.set_title("Thrust vs Time")
        thrustAx.set_xlabel("Time (s)")
        thrustAx.set_ylabel("Thrust (kN)")
        thrustAx.legend()
        thrustAx.grid()
        thrustAx.axhline(-self.params.T_max, color="red", linestyle="--")
        thrustAx.axhline(self.params.T_max, color="red", linestyle="--")

        xyMin = min(xmin, ymin)
        xyMax = max(xmax, ymax)
        x = np.linspace(xyMin, xyMax, 10)
        y = np.linspace(xyMin, xyMax, 10)
        X, Y = np.meshgrid(x, y)
        Z = (
            np.sin(self.params.max_glide_slope)
            * np.sqrt((X - self.params.xf[0]) ** 2 + (Y - self.params.xf[1]) ** 2)
            + self.params.xf[2]
        )
        label = {"label": "Glide Cone"} if withLabels else {}
        threeDimAx.plot_surface(X, Y, Z, alpha=0.3, color="gray", **label)
        threeDimAx.set_title("3D Trajectory")
        threeDimAx.set_xlabel("X (km)")
        threeDimAx.set_ylabel("Y (km)")
        threeDimAx.set_zlabel("Z (km)")
        threeDimAx.legend()
        threeDimAx.grid()
        # threeDimAx.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))

        plt.tight_layout()
        if show:
            plt.show()
