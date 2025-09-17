from System import SystemParameters, Array3

from typing import Union

import pyomo.kernel as pmo
from pyomo.core.kernel import conic
import numpy as np
from PyomoTools.kernel.Formulations import Conic


class State:
    def __init__(
        self,
        mass: Union[float, pmo.variable],
        position: Array3,
        velocity: Array3,
        acceleration: Array3,
        thrust: Array3,
        gamma: Union[float, pmo.variable],
    ):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.thrust = thrust
        self.gamma = gamma


class LosslessConvexification_Step_Model(pmo.block):
    """
    This model, even with its convexified constraints, is still highly nonconvex. Therefore, standard interior-point solvers (e.g., ipopt) still converge to infeasible stationary points.
    """

    X = 0
    Y = 1
    Z = 2

    def __init__(
        self,
        params: SystemParameters,
        dt: Union[float, pmo.variable],  # In the paper, dt is actually variable.
        prevState: State = None,
        isFinal: bool = False,
    ):
        super().__init__()

        self.params = params
        self._dt = (
            lambda: dt
        )  # Loose reference to dt allows this step model to not assume ownership of dt variable
        self.math = params.math
        self.prevState = prevState

        self.variables()
        self.constraints()
        if isFinal:
            self.final_constraints()

    @property
    def dt(self):
        return self._dt()

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

        self.velocity = pmo.variable_list(
            [
                pmo.variable(value=pmo.value(self.prevState.velocity[i]))
                for i in range(3)
            ]
        )

        self.acceleration = pmo.variable_list(
            [pmo.variable(value=0.0) for _ in range(3)]
        )

        self.thrust = pmo.variable_list(
            [pmo.variable(value=pmo.value(self.prevState.thrust[i])) for i in range(3)]
        )
        self.gamma = pmo.variable(
            domain=pmo.NonNegativeReals,
            value=pmo.value(self.prevState.gamma),
        )

        df = self.params.ComputeDragForce(self.velocity)

        self.drag_force = pmo.variable_list(
            [pmo.variable(value=pmo.value(df[i])) for i in range(3)]
        )

    def constraints(self):
        # Linear Constraint (constant dt)
        # Bilinear, Non-convex Constraint (variable dt)
        self.mass_evolution = pmo.constraint(
            self.mass
            == self.prevState.mass
            - self.dt
            * (
                self.params.alpha / 2 * (self.gamma + self.prevState.gamma)
                + self.params.mdot_bp
            )
        )

        self.position_evolution = pmo.constraint_list()
        self.velocity_evolution = pmo.constraint_list()
        self.newtons_second_law = pmo.constraint_list()
        self.drag_force_def = pmo.constraint_list()

        df = self.params.ComputeDragForce(self.velocity)

        for i in range(3):
            # Linear Constraint (constant dt)
            # Nonlinear, Non-convex Constraint (variable dt)
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

            # Linear Constraint (constant dt)
            # Bilinear, Non-convex Constraint (variable dt)
            self.velocity_evolution.append(
                pmo.constraint(
                    self.velocity[i]
                    == self.prevState.velocity[i]
                    + (self.acceleration[i] + self.prevState.acceleration[i])
                    / 2
                    * self.dt
                )
            )

            # Bilinear, Non-convex Constraint
            self.newtons_second_law.append(
                pmo.constraint(
                    self.mass * self.acceleration[i]
                    == self.thrust[i]
                    + self.drag_force[i]
                    + self.params.g[i] * self.mass
                )
            )

            # Nonlinear, Non-convex Constraint
            self.drag_force_def.append(pmo.constraint(self.drag_force[i] == df[i]))

        # Nonlinear, Convex Constraint
        self.gamma_def = Conic(r=self.gamma, x=self.thrust, order=2)

        # Nonlinear, Convex Constraint
        self.glide_slope = Conic(
            r=self.math.dot(self.params.e_u, self.position)
            / np.cos(self.params.max_glide_slope),
            x=self.position,
            order=2,
        )

        # Linear Constraint
        self.thrust_lower_limit = pmo.constraint(self.gamma >= self.params.T_min)

        # Linear Constraint
        self.thrust_upper_limit = pmo.constraint(self.gamma <= self.params.T_max)

        # Linear Constraint
        self.tilt_angle = pmo.constraint(
            self.math.dot(self.params.e_u, self.thrust)
            >= np.cos(self.params.max_tilt) * self.gamma
        )

        # Linear Constraint
        self.trust_ramp_upper_limit = pmo.constraint(
            (self.gamma - self.prevState.gamma) <= self.params.dTdt_max * self.dt
        )

        # Linear Constraint
        self.trust_ramp_lower_limit = pmo.constraint(
            (self.gamma - self.prevState.gamma) >= self.params.dTdt_min * self.dt
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
                pmo.constraint(self.thrust[i] == self.gamma * self.params.nf[i])
            )

    def getState(self) -> State:
        return State(
            mass=self.mass,
            position=self.position,
            velocity=self.velocity,
            acceleration=self.acceleration,
            thrust=self.thrust,
            gamma=self.gamma,
        )


class LosslessConvexificationModel(pmo.block):
    def __init__(
        self, params: SystemParameters, nSteps: int, start: float, stop: float = None
    ):
        super().__init__()
        self.params = params
        self.start = start
        self.nSteps = nSteps

        if stop is None:
            self._stop = stop
            self.variable_dt = True
            self.dt = pmo.variable(lb=0.5, ub=1000, value=1.0)
        else:
            self.variable_dt = False
            self._stop = stop
            self.dt = (stop - start) / nSteps

        self.initState = State(
            mass=params.m0,
            position=params.x0,
            velocity=params.v0,
            acceleration=[0.0, 0.0, 0.0],
            thrust=params.T0,
            gamma=np.linalg.norm(params.T0),
        )

        self.steps = pmo.block_list()
        for i in range(nSteps):
            isFinal = i == nSteps - 1
            prevState = self.initState if i == 0 else self.steps[i - 1].getState()
            step = LosslessConvexification_Step_Model(
                params=params, dt=self.dt, prevState=prevState, isFinal=isFinal
            )
            self.steps.append(step)

    @property
    def stop(self):
        if self.variable_dt:
            return self.start + self.nSteps * self.dt
        else:
            return self._stop
