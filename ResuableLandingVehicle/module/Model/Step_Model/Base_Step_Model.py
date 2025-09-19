"""
Base_Step_Model
===============

This module defines the `Base_Step_Model` class, which serves as an abstract base class for modeling a single step in a simulation.
It provides methods for defining variables, constraints, and evolution functions for mass, position, and velocity.
Subclasses must implement abstract methods to define specific behaviors.
"""

from ..State import InitializationState, State, IterationState
from ...System import SystemParameters
from ...Util.Math import Array3

from abc import ABC, abstractmethod
from typing import Union, Dict, Any
import numpy as np
import pyomo.kernel as pmo
from functools import cached_property

from PyomoTools.kernel.Formulations import Conic


class Base_Step_Model(pmo.block, ABC):
    """
    Base_Step_Model
    ----------------

    Abstract base class for modeling a single step in a simulation.

    Attributes:
        t_est (float): Estimated time for determining wind speed.
        params (SystemParameters): System parameters for the simulation.
        dt (Union[float, pmo.variable]): Time step duration.
        prevState (State): State of the system in the previous step.
        isFinal (bool): Indicates whether this is the final step.
        otherParams (Dict[str, Any]): Additional parameters for the step model.
        initializationState (InitializationState): Initial state of the system.
    """

    X = 0
    Y = 1
    Z = 2

    def __init__(
        self,
        t_est: float,  # Estimated time (used for determining wind speed)
        params: SystemParameters,
        dt: Union[float, pmo.variable],
        prevState: State = None,
        isFinal: bool = False,
        otherParams: Dict[str, Any] = {},
        initializationState: InitializationState = None,
    ):
        """
        Initializes the `Base_Step_Model` with the given parameters.

        Args:
            t_est (float): Estimated time for determining wind speed.
            params (SystemParameters): System parameters for the simulation.
            dt (Union[float, pmo.variable]): Time step duration.
            prevState (State, optional): State of the system in the previous step. Defaults to None.
            isFinal (bool, optional): Indicates whether this is the final step. Defaults to False.
            otherParams (Dict[str, Any], optional): Additional parameters for the step model. Defaults to {}.
            initializationState (InitializationState, optional): Initial state of the system. Defaults to None.
        """
        super().__init__()

        self.t_est = t_est
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
        self.isFinal = isFinal
        if isFinal:
            self.final_constraints()

    @property
    def dt(self):
        """
        Retrieves the time step duration.

        Returns:
            Union[float, pmo.variable]: Time step duration.
        """
        return self._dt()

    def variables(self, initializationState: InitializationState = None):
        """
        Defines the variables for the step model, including mass, position, velocity, acceleration, and thrust.

        Args:
            initializationState (InitializationState, optional): Initial state of the system. Defaults to None.
        """

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
        """
        Computes the base mass evolution function.

        Args:
            dt: Time step duration.
            gamma: Current thrust magnitude.
            prev_gamma: Previous thrust magnitude.

        Returns:
            float: Change in mass over the time step.
        """
        # s * ((Mg/s)/kN * (kN) + Mg/s) = Mg
        return -dt * (
            self.params.alpha / 2 * (gamma + prev_gamma) + self.params.mdot_bp
        )  # Mg

    @abstractmethod
    def mass_evolution_function(self, dt, gamma, prev_gamma):
        """
        Abstract method for computing the mass evolution function.

        Args:
            dt: Time step duration.
            gamma: Current thrust magnitude.
            prev_gamma: Previous thrust magnitude.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement mass_evolution_function method"
        )

    def base_position_evolution_function(self, dt, prev_vel_i, accel_i, prev_accel_i):
        """
        Computes the base position evolution function.

        Args:
            dt: Time step duration.
            prev_vel_i: Previous velocity component.
            accel_i: Current acceleration component.
            prev_accel_i: Previous acceleration component.

        Returns:
            float: Change in position over the time step.
        """
        # s * (km/s) + (km/s^2 * s^2) = km
        return dt * prev_vel_i + (accel_i + prev_accel_i / 2) / 3 * dt**2

    @abstractmethod
    def position_evolution_function(self, dt, prev_vel_i, accel_i, prev_accel_i):
        """
        Abstract method for computing the position evolution function.

        Args:
            dt: Time step duration.
            prev_vel_i: Previous velocity component.
            accel_i: Current acceleration component.
            prev_accel_i: Previous acceleration component.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement position_evolution_function method"
        )

    def base_velocity_evolution_function(self, dt, accel_i, prev_accel_i):
        """
        Computes the base velocity evolution function.

        Args:
            dt: Time step duration.
            accel_i: Current acceleration component.
            prev_accel_i: Previous acceleration component.

        Returns:
            float: Change in velocity over the time step.
        """
        # s * (km/s^2) = km/s
        return (accel_i + prev_accel_i) / 2 * dt

    @abstractmethod
    def velocity_evolution_function(self, dt, accel_i, prev_accel_i):
        """
        Abstract method for computing the velocity evolution function.

        Args:
            dt: Time step duration.
            accel_i: Current acceleration component.
            prev_accel_i: Previous acceleration component.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement velocity_evolution_function method"
        )

    @abstractmethod
    def ComputeDragForce(self) -> Array3:
        """
        Abstract method for computing the drag force.

        Returns:
            Array3: Computed drag force.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement ComputeDragForce method")

    @cached_property
    def wind_velocity(self) -> Array3:
        """
        Computes the wind velocity at the estimated time.

        Returns:
            Array3: Computed wind velocity.
        """
        return self.params.wind_function(self.t_est)[0]

    @abstractmethod
    def NewtonsSecondLaw(self, i):
        """
        Abstract method for computing Newton's second law.

        Args:
            i: Index for the computation.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement NewtonsSecondLaw method")

    def constraints(self):
        """
        Defines the constraints for the step model, including mass evolution, position evolution, and velocity evolution.
        """
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
        """
        Defines the final constraints for the step model, including position, velocity, and direction constraints.
        """
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
        """
        Retrieves the current state of the system.

        Returns:
            State: Current state of the system.
        """
        return State(
            mass=self.mass,
            position=self.position,
            velocity=self.velocity,
            acceleration=self.acceleration,
            thrust=self.thrust,
            gamma=self.gamma,
        )

    def getIterationState(self) -> IterationState:
        """
        Retrieves the iteration state of the system.

        Returns:
            IterationState: Iteration state of the system.
        """
        return IterationState(
            dt=pmo.value(self.dt),
            mass=pmo.value(self.mass),
            prev_mass=pmo.value(self.prevState.mass),
            gamma=pmo.value(self.gamma),
            prev_gamma=pmo.value(self.prevState.gamma),
            velocity=[pmo.value(v) for v in self.velocity],
            prev_velocity=[pmo.value(v) for v in self.prevState.velocity],
            wind_velocity=self.wind_velocity,
            thrust=[pmo.value(t) for t in self.thrust],
            prev_thrust=[pmo.value(t) for t in self.prevState.thrust],
            acceleration=[pmo.value(a) for a in self.acceleration],
            prev_acceleration=[pmo.value(a) for a in self.prevState.acceleration],
            position=[pmo.value(p) for p in self.position],
            artificial_acceleration=[
                pmo.value(a) for a in self.artificial_acceleration
            ],
        )
