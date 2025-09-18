from ...System import SystemParameters
from ..State import State, IterationState

import pyomo.kernel as pmo
from abc import ABC
from typing import Union, Iterable
import numpy as np


class Base_Model(pmo.block, ABC):
    def __init__(
        self,
        params: SystemParameters,
        nSteps: int,
        start: float,
        stop: Union[float, None],
    ):
        super().__init__()
        self.params = params
        self.start = start
        self.nSteps = nSteps

        if stop is None:
            self.variable_dt = True
            self.dt = pmo.variable(domain=pmo.NonNegativeReals, value=1.0)
            self.stop = self.start + self.nSteps * self.dt
        else:
            self.variable_dt = False
            self.dt = (stop - start) / nSteps
            self.stop = stop

        self.initState = State(
            mass=params.m0,
            position=params.x0,
            velocity=params.v0,
            acceleration=[0.0, 0.0, 0.0],
            thrust=params.T0,
            gamma=np.linalg.norm(params.T0),
        )

    def getIterationStates(self) -> Iterable[IterationState]:
        return [step.getIterationState() for step in self.steps]

    def Plot(self):
        # import matplotlib

        # matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        dt = pmo.value(self.dt)
        times = [self.start + i * dt for i in range(self.nSteps)]
        positions = np.array(
            [[pmo.value(step.position[i]) for step in self.steps] for i in range(3)]
        )
        velocities = np.array(
            [[pmo.value(step.velocity[i]) for step in self.steps] for i in range(3)]
        )
        accelerations = np.array(
            [[pmo.value(step.acceleration[i]) for step in self.steps] for i in range(3)]
        )
        masses = np.array([pmo.value(step.mass) for step in self.steps])

        fig = plt.figure(figsize=(12, 16))
        gs = fig.add_gridspec(3, 2)
        posAx = fig.add_subplot(gs[0, 0])
        velAx = fig.add_subplot(gs[0, 1])
        accAx = fig.add_subplot(gs[1, 0])
        massAx = fig.add_subplot(gs[1, 1])
        thrustAx = fig.add_subplot(gs[2, 0])

        threeDimAx = fig.add_subplot(gs[2, 1], projection="3d")
        posAx.plot(times, positions[0, :], label="X")
        posAx.plot(times, positions[1, :], label="Y")
        posAx.plot(times, positions[2, :], label="Z")
        posAx.set_title("Position vs Time")
        posAx.set_xlabel("Time (s)")
        posAx.set_ylabel("Position (km)")
        posAx.legend()
        posAx.grid()

        velAx.plot(times, velocities[0, :], label="Vx")
        velAx.plot(times, velocities[1, :], label="Vy")
        velAx.plot(times, velocities[2, :], label="Vz")
        velAx.set_title("Velocity vs Time")
        velAx.set_xlabel("Time (s)")
        velAx.set_ylabel("Velocity (km/s)")
        velAx.legend()
        velAx.grid()

        accAx.plot(times, accelerations[0, :], label="Ax")
        accAx.plot(times, accelerations[1, :], label="Ay")
        accAx.plot(times, accelerations[2, :], label="Az")
        accAx.set_title("Acceleration vs Time")
        accAx.set_xlabel("Time (s)")
        accAx.set_ylabel("Acceleration (km/s²)")
        accAx.legend()
        accAx.grid()

        massAx.plot(times, masses, label="Mass", color="purple")
        massAx.axhline(self.params.m_dry, color="red", linestyle="--", label="Dry Mass")
        massAx.set_title("Mass vs Time")
        massAx.set_xlabel("Time (s)")
        massAx.set_ylabel("Mass (Mg)")
        massAx.legend()
        massAx.grid()

        thrusts = np.array(
            [[pmo.value(step.thrust[i]) for step in self.steps] for i in range(3)]
        )
        thrustAx.plot(times, thrusts[0, :], label="Tx")
        thrustAx.plot(times, thrusts[1, :], label="Ty")
        thrustAx.plot(times, thrusts[2, :], label="Tz")
        thrustAx.set_title("Thrust vs Time")
        thrustAx.set_xlabel("Time (s)")
        thrustAx.set_ylabel("Thrust (kN)")
        thrustAx.legend()
        thrustAx.grid()
        thrustAx.axhline(-self.params.T_max, color="red", linestyle="--")
        thrustAx.axhline(self.params.T_max, color="red", linestyle="--")

        threeDimAx.plot(
            positions[0, :], positions[1, :], positions[2, :], label="Trajectory"
        )
        nFigurines = 10
        figurineIndices = np.linspace(0, self.nSteps - 1, nFigurines).astype(int)
        for idx in figurineIndices:
            if idx == 0:
                kwargs = dict(label="Thrust Vector")
            else:
                kwargs = {}
            threeDimAx.quiver(
                positions[0, idx],
                positions[1, idx],
                positions[2, idx],
                thrusts[0, idx],
                thrusts[1, idx],
                thrusts[2, idx],
                length=np.linalg.norm(thrusts[:, idx]) / 500,
                normalize=True,
                color="red",
                pivot="middle",
                arrow_length_ratio=0.007,
                **kwargs,
            )

        xmin = min(min(positions[0, :]), self.params.xf[0])
        xmax = max(max(positions[0, :]), self.params.xf[0])
        ymin = min(min(positions[1, :]), self.params.xf[1])
        ymax = max(max(positions[1, :]), self.params.xf[1])
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
        threeDimAx.plot_surface(X, Y, Z, alpha=0.3, color="gray", label="Glide Cone")
        threeDimAx.set_title("3D Trajectory")
        threeDimAx.set_xlabel("X (km)")
        threeDimAx.set_ylabel("Y (km)")
        threeDimAx.set_zlabel("Z (km)")
        threeDimAx.legend()
        threeDimAx.grid()
        # threeDimAx.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))

        plt.tight_layout()
        plt.show()
