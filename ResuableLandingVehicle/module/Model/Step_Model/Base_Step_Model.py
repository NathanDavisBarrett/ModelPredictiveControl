from ..State import InitializationState, State, IterationState
from ...System import SystemParameters
from ...Util.Math import Array3

from abc import ABC, abstractmethod
from typing import Union, Dict, Any
import numpy as np
import pyomo.kernel as pmo

from PyomoTools.kernel.Formulations import Conic


class Base_Step_Model(pmo.block, ABC):
    X = 0
    Y = 1
    Z = 2

    def __init__(
        self,
        params: SystemParameters,
        dt: Union[float, pmo.variable],
        prevState: State = None,
        isFinal: bool = False,
        otherParams: Dict[str, Any] = {},
        initializationState: InitializationState = None,
    ):
        super().__init__()

        self.params = params
        for key, value in otherParams.items():
            setattr(self, key, value)
        self._dt = (
            lambda: dt
        )  # Keeping this as a function allows for referencing without this step model trying to take ownership of the variable.
        self.math = params.math
        self.prevState = prevState

        self.variables(initializationState=initializationState)
        self.constraints()
        if isFinal:
            self.final_constraints()

    @property
    def dt(self):
        return self._dt()

    def variables(self, initializationState: InitializationState = None):

        mass_val = (
            pmo.value(self.prevState.mass)
            if initializationState is None
            else initializationState.mass
        )
        self.mass = pmo.variable(lb=self.params.m_dry, value=mass_val)

        poss_val = (
            [pmo.value(self.prevState.position[i]) for i in range(3)]
            if initializationState is None
            else initializationState.position
        )
        self.position = pmo.variable_list(
            [pmo.variable(value=poss_val[i]) for i in range(3)]
        )

        vel_val = (
            [pmo.value(self.prevState.velocity[i]) for i in range(3)]
            if initializationState is None
            else initializationState.velocity
        )
        if np.allclose(vel_val, [0.0, 0.0, 0.0]):
            vel_val = [1e-5, 1e-5, 1e-5]  # Avoid numerical issues with zero velocity
        self.velocity = pmo.variable_list(
            [pmo.variable(value=vel_val[i]) for i in range(3)]
        )

        accel_val = (
            [pmo.value(self.prevState.acceleration[i]) for i in range(3)]
            if initializationState is None
            else initializationState.acceleration
        )
        self.acceleration = pmo.variable_list(
            [pmo.variable(value=accel_val[i]) for i in range(3)]
        )

        art_accel_val = (
            [0.0, 0.0, 0.0]
            if initializationState is None
            else initializationState.artificial_acceleration
        )
        self.artificial_acceleration = pmo.variable_list(
            [pmo.variable(value=art_accel_val[i]) for i in range(3)]
        )

        art_accel_mag_val = (
            1e-5
            if initializationState is None
            else initializationState.artificial_acceleration_mag
        )
        self.artificial_acceleration_mag = pmo.variable(
            domain=pmo.NonNegativeReals, value=art_accel_mag_val
        )

        thrust_val = (
            [pmo.value(self.prevState.thrust[i]) for i in range(3)]
            if initializationState is None
            else initializationState.thrust
        )
        self.thrust = pmo.variable_list(
            [pmo.variable(value=thrust_val[i]) for i in range(3)]
        )

        gamma_val = (
            pmo.value(self.prevState.gamma)
            if initializationState is None
            else initializationState.gamma
        )
        self.gamma = pmo.variable(
            domain=pmo.NonNegativeReals,
            value=gamma_val,
        )

        df = self.ComputeDragForce()

        self.drag_force = pmo.variable_list(
            [pmo.variable(value=pmo.value(df[i])) for i in range(3)]
        )

    def base_mass_evolution_function(self, dt, gamma, prev_gamma):
        # s * ((Mg/s)/kN * (kN) + Mg/s) = Mg
        return -dt * (
            self.params.alpha / 2 * (gamma + prev_gamma) + self.params.mdot_bp
        )  # Mg

    @abstractmethod
    def mass_evolution_function(self, dt, gamma, prev_gamma):
        raise NotImplementedError(
            "Subclasses must implement mass_evolution_function method"
        )

    def base_position_evolution_function(self, dt, prev_vel_i, accel_i, prev_accel_i):
        # s * (km/s) + (km/s^2 * s^2) = km
        return dt * prev_vel_i + (accel_i + prev_accel_i / 2) / 3 * dt**2

    @abstractmethod
    def position_evolution_function(self, dt, prev_vel_i, accel_i, prev_accel_i):
        raise NotImplementedError(
            "Subclasses must implement position_evolution_function method"
        )

    def base_velocity_evolution_function(self, dt, accel_i, prev_accel_i):
        # s * (km/s^2) = km/s
        return (accel_i + prev_accel_i) / 2 * dt

    @abstractmethod
    def velocity_evolution_function(self, dt, accel_i, prev_accel_i):
        raise NotImplementedError(
            "Subclasses must implement velocity_evolution_function method"
        )

    @abstractmethod
    def ComputeDragForce(self) -> Array3:
        raise NotImplementedError("Subclasses must implement ComputeDragForce method")

    @abstractmethod
    def NewtonsSecondLaw(self, i):
        raise NotImplementedError("Subclasses must implement NewtonsSecondLaw method")

    def constraints(self):
        # Linear Constraint (constant dt)
        self.mass_evolution = pmo.constraint(
            self.mass - self.prevState.mass
            == self.mass_evolution_function(self.dt, self.gamma, self.prevState.gamma)
        )

        self.position_evolution = pmo.constraint_list()
        self.velocity_evolution = pmo.constraint_list()
        self.newtons_second_law = pmo.constraint_list()
        self.drag_force_def = pmo.constraint_list()

        df = self.ComputeDragForce()

        for i in range(3):
            # Linear Constraint (constant dt)
            self.position_evolution.append(
                pmo.constraint(
                    self.position[i] - self.prevState.position[i]
                    == self.position_evolution_function(
                        self.dt,
                        self.prevState.velocity[i],
                        self.acceleration[i],
                        self.prevState.acceleration[i],
                        i,
                    )
                )
            )

            # Linear Constraint (constant dt)
            self.velocity_evolution.append(
                pmo.constraint(
                    self.velocity[i] - self.prevState.velocity[i]
                    == self.velocity_evolution_function(
                        self.dt, self.acceleration[i], self.prevState.acceleration[i], i
                    )
                )
            )

            # Linear Constraint
            self.newtons_second_law.append(pmo.constraint(self.NewtonsSecondLaw(i)))

            # Linear Constraint
            self.drag_force_def.append(pmo.constraint(self.drag_force[i] == df[i]))

        # Nonlinear, Convex Constraint
        self.artificial_acceleration_mag_def = Conic(
            r=self.artificial_acceleration_mag,
            x=self.artificial_acceleration,
            order=2,
        )

        # Nonlinear, Convex Constraint
        self.gamma_def = Conic(r=self.gamma, x=self.thrust, order=2)

        # Nonlinear, Convex Constraint
        relative_position = self.math.vector_add(
            self.position, self.math.vector_scale(self.params.xf, -1)
        )
        self.glide_slope = Conic(
            r=self.math.dot(self.params.e_u, relative_position)
            / np.cos(self.params.max_glide_slope),
            x=relative_position,
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
        )  # kN = (kN/s) * s

        # Linear Constraint
        self.trust_ramp_lower_limit = pmo.constraint(
            (self.gamma - self.prevState.gamma) >= self.params.dTdt_min * self.dt
        )  # kN = (kN/s) * s

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

    def getIterationState(self) -> IterationState:
        return IterationState(
            dt=pmo.value(self.dt),
            mass=pmo.value(self.mass),
            prev_mass=pmo.value(self.prevState.mass),
            gamma=pmo.value(self.gamma),
            prev_gamma=pmo.value(self.prevState.gamma),
            velocity=[pmo.value(v) for v in self.velocity],
            prev_velocity=[pmo.value(v) for v in self.prevState.velocity],
            thrust=[pmo.value(t) for t in self.thrust],
            prev_thrust=[pmo.value(t) for t in self.prevState.thrust],
            acceleration=[pmo.value(a) for a in self.acceleration],
            prev_acceleration=[pmo.value(a) for a in self.prevState.acceleration],
            position=[pmo.value(p) for p in self.position],
            artificial_acceleration=[
                pmo.value(a) for a in self.artificial_acceleration
            ],
        )
