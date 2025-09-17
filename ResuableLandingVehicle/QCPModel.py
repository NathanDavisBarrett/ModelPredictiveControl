from System import SystemParameters, Array3

from typing import Union

import pyomo.kernel as pmo
import numpy as np


class State:
    def __init__(
        self,
        mass: Union[float, pmo.variable],
        position: Array3,
        velocity: Array3,
        acceleration: Array3,
        thrust: Array3,
        thrust_magnitude: Union[float, pmo.variable],
    ):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.thrust = thrust
        self.thrust_magnitude = thrust_magnitude


class QCP_Step_Model(pmo.block):
    """
    The non-convex problem is highly non-convex. Standard interior point solvers (like ipopt) converge to an answer quickly and reliably, but the solution they converge to is almost always infeasible. My goal with this model was to reformulate the problem as a Quadratically Constraint Program (QCP) and solve it with a solver like Gurobi that is designed to handle these types of problems. However, in doing so, I had to introduce even more stark non-convexities. This made it such that not even gurobi could find a feasible solution within a reasonable item frame.
    """

    X = 0
    Y = 1
    Z = 2

    def __init__(
        self,
        params: SystemParameters,
        dt: float,
        prevState: State = None,
        isFinal: bool = False,
    ):
        super().__init__()

        self.params = params
        self.dt = dt
        self.math = params.math
        self.prevState = prevState

        self.variables()
        self.constraints()
        if isFinal:
            self.final_constraints()

    def variables(self):
        self.mass = pmo.variable(
            lb=self.params.m_dry, value=pmo.value(self.prevState.mass)
        )
        self.position = pmo.variable_list(
            [
                pmo.variable(value=pmo.value(self.prevState.position[i]))
                for i in range(3)
            ]
        )
        self.postition_magnitude = pmo.variable(
            domain=pmo.NonNegativeReals,
            value=self.math.norm([pmo.value(self.position[i]) for i in range(3)]),
        )

        self.velocity = pmo.variable_list(
            [
                pmo.variable(value=pmo.value(self.prevState.velocity[i]))
                for i in range(3)
            ]
        )
        self.velocity_sq = pmo.variable_list(
            [pmo.variable(value=pmo.value(self.velocity[i]) ** 2) for i in range(3)]
        )
        self.velocity_mag_sq = pmo.variable(
            domain=pmo.NonNegativeReals,
            value=self.math.dot(
                [pmo.value(self.velocity[i]) for i in range(3)],
                [pmo.value(self.velocity[i]) for i in range(3)],
            ),
        )

        self.acceleration = pmo.variable_list(
            [pmo.variable(value=0.0) for _ in range(3)]
        )

        self.thrust = pmo.variable_list(
            [pmo.variable(value=pmo.value(self.prevState.thrust[i])) for i in range(3)]
        )
        self.thrust_magnitude = pmo.variable(
            domain=pmo.NonNegativeReals,
            value=self.math.norm([pmo.value(self.thrust[i]) for i in range(3)]),
        )

        df = self.params.ComputeDragForce(self.velocity)

        self.drag_force = pmo.variable_list(
            [pmo.variable(value=pmo.value(df[i])) for i in range(3)]
        )

    def constraints(self):
        self.mass_evolution = pmo.constraint(
            self.mass
            == self.prevState.mass
            - self.dt
            * (
                self.params.alpha
                / 2
                * (self.thrust_magnitude + self.prevState.thrust_magnitude)
                + self.params.mdot_bp
            )
        )

        self.position_evolution = pmo.constraint_list()
        self.velocity_evolution = pmo.constraint_list()
        self.velocity_sq_def = pmo.constraint_list()
        self.newtons_second_law = pmo.constraint_list()
        self.drag_force_def = pmo.constraint_list()
        self.drag_force_colinearity = pmo.constraint_list()

        drag_force_sq = self.params.ComputeDragForceSq(
            self.velocity_mag_sq, self.velocity_sq
        )
        drag_vel_cross = self.math.cross(
            self.drag_force, self.math.vector_scale(self.velocity, -1)
        )

        for i in range(3):
            self.position_evolution.append(
                pmo.constraint(
                    self.position[i]
                    == self.prevState.position[i]
                    + self.dt * self.prevState.velocity[i]
                    + (self.acceleration[i] + self.prevState.acceleration[i] / 2)
                    / 3
                    * self.dt**2
                )
            )

            self.velocity_evolution.append(
                pmo.constraint(
                    self.velocity[i]
                    == self.prevState.velocity[i]
                    + (self.acceleration[i] + self.prevState.acceleration[i])
                    / 2
                    * self.dt
                )
            )

            self.velocity_sq_def.append(
                pmo.constraint(self.velocity_sq[i] == self.velocity[i] ** 2)
            )

            self.newtons_second_law.append(
                pmo.constraint(
                    self.mass * self.acceleration[i]
                    == self.thrust[i]
                    + self.drag_force[i]
                    + self.params.g[i] * self.mass
                )
            )

            self.drag_force_def.append(
                pmo.constraint(self.drag_force[i] ** 2 == drag_force_sq[i])
            )

            self.drag_force_colinearity.append(pmo.constraint(drag_vel_cross[i] == 0))

        self.velocity_mag_sq_def = pmo.constraint(
            self.velocity_mag_sq == sum(self.velocity_sq[i] for i in range(3))
        )

        self.drag_force_direction = pmo.constraint(
            self.math.dot(self.drag_force, self.math.vector_scale(self.velocity, -1))
            >= 0
        )

        self.position_magnitude_def = pmo.constraint(
            self.postition_magnitude**2 == sum(self.position[i] ** 2 for i in range(3))
        )

        self.thrust_magnitude_def = pmo.constraint(
            self.thrust_magnitude**2 == sum(self.thrust[i] ** 2 for i in range(3))
        )

        self.glide_slope = pmo.constraint(
            self.postition_magnitude * np.cos(self.params.max_glide_slope)
            <= self.math.dot(self.params.e_u, self.position)
        )

        self.thrust_lower_limit = pmo.constraint(
            self.thrust_magnitude >= self.params.T_min
        )
        self.thrust_upper_limit = pmo.constraint(
            self.thrust_magnitude <= self.params.T_max
        )

        self.tilt_angle = pmo.constraint(
            self.math.dot(self.params.e_u, self.thrust)
            >= np.cos(self.params.max_tilt) * self.thrust_magnitude
        )

        self.trust_ramp_upper_limit = pmo.constraint(
            (self.thrust_magnitude - self.prevState.thrust_magnitude) / self.dt
            <= self.params.dTdt_max
        )
        self.trust_ramp_lower_limit = pmo.constraint(
            (self.thrust_magnitude - self.prevState.thrust_magnitude) / self.dt
            >= self.params.dTdt_min
        )

    def final_constraints(self):
        self.final_position = pmo.constraint_list()
        self.final_velocity = pmo.constraint_list()
        self.final_direction = pmo.constraint_list()
        for i in range(3):
            self.final_position.append(
                pmo.constraint(self.position[i] == self.params.xf[i])
            )
            self.final_velocity.append(
                pmo.constraint(self.velocity[i] == self.params.vf[i])
            )
            self.final_direction.append(
                pmo.constraint(
                    self.thrust[i] == self.thrust_magnitude * self.params.nf[i]
                )
            )

    def getState(self) -> State:
        return State(
            mass=self.mass,
            position=self.position,
            velocity=self.velocity,
            acceleration=self.acceleration,
            thrust=self.thrust,
            thrust_magnitude=self.thrust_magnitude,
        )


class QCPModel(pmo.block):
    def __init__(
        self, params: SystemParameters, start: float, stop: float, nSteps: int
    ):
        super().__init__()
        self.params = params
        self.start = start
        self.stop = stop
        self.nSteps = nSteps
        self.dt = (stop - start) / nSteps

        self.initState = State(
            mass=params.m0,
            position=params.x0,
            velocity=params.v0,
            acceleration=[0.0, 0.0, 0.0],
            thrust=params.T0,
            thrust_magnitude=np.linalg.norm(params.T0),
        )

        self.steps = pmo.block_list()
        for i in range(nSteps):
            isFinal = i == nSteps - 1
            prevState = self.initState if i == 0 else self.steps[i - 1].getState()
            step = QCP_Step_Model(
                params=params, dt=self.dt, prevState=prevState, isFinal=isFinal
            )
            self.steps.append(step)
