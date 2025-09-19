"""
Iterate_Step_Model
==================

This module defines the `Iterate_Step_Model` class, which extends the `Base_Step_Model` to provide specific implementations for iterative steps in a simulation.
"""

from .Base_Step_Model import Base_Step_Model
from ...System import SystemParameters
from ...Util.Math import Array3
from ..State import State, InitializationState, IterationState

import pyomo.kernel as pmo
import numpy as np
from typing import Union


class Iterate_Step_Model(Base_Step_Model):
    """
    Iterate_Step_Model
    ------------------

    Extends the `Base_Step_Model` to provide specific implementations for iterative steps in a simulation.

    Attributes:
        t_est (float): Estimated time for determining wind speed.
        params (SystemParameters): System parameters for the simulation.
        dt (Union[float, pmo.variable]): Time step duration.
        prevTimeState (State): State of the system in the previous time step.
        prevIterationState (IterationState): State of the system in the previous iteration.
        isFinal (bool): Indicates whether this is the final step.
    """

    def __init__(
        self,
        t_est: float,
        params: SystemParameters,
        dt: Union[float, pmo.variable],
        prevTimeState: State,
        prevIterationState: IterationState,
        isFinal: bool = False,
    ):
        """
        Initializes the `Iterate_Step_Model` with the given parameters.

        Args:
            t_est (float): Estimated time for determining wind speed.
            params (SystemParameters): System parameters for the simulation.
            dt (Union[float, pmo.variable]): Time step duration.
            prevTimeState (State): State of the system in the previous time step.
            prevIterationState (IterationState): State of the system in the previous iteration.
            isFinal (bool, optional): Indicates whether this is the final step. Defaults to False.
        """

        super().__init__(
            t_est=t_est,
            params=params,
            dt=dt,
            prevState=prevTimeState,
            isFinal=isFinal,
            otherParams=dict(
                prevTimeState=prevTimeState, prevIterationState=prevIterationState
            ),
            initializationState=prevIterationState,
        )

    def variables(self, initializationState: InitializationState = None):
        """
        Defines the variables for the iterative step model, including thrust change and eta_thrust.

        Args:
            initializationState (InitializationState, optional): Initial state of the system. Defaults to None.
        """
        super().variables(initializationState=initializationState)

        thrust_change = self.math.vector_add(
            initializationState.thrust,
            self.math.vector_scale(self.prevIterationState.prev_thrust, -1),
        )

        eta_val = self.math.dot(thrust_change, thrust_change)

        self.eta_thrust = pmo.variable(domain=pmo.NonNegativeReals, value=eta_val)

    def constraints(self):
        """
        Defines the constraints for the iterative step model, including thrust change constraints.
        """
        super().constraints()

        # Nonlinear, Convex Constraint
        thrust_change = self.math.vector_add(
            self.thrust, self.math.vector_scale(self.prevIterationState.prev_thrust, -1)
        )
        self.eta_thrust_def = pmo.constraint(
            self.math.dot(thrust_change, thrust_change) <= self.eta_thrust
        )

    def base_mass_evolution_derivative(self, dt, gamma, prev_gamma):
        """
        Computes the derivative of the base mass evolution function.

        Args:
            dt (float): Time step duration.
            gamma (float): Current thrust magnitude.
            prev_gamma (float): Previous thrust magnitude.

        Returns:
            List[float]: Derivatives of the mass evolution function with respect to dt, gamma, and prev_gamma.
        """
        # 0: (Mg/s)/kN * kN - Mg/s = Mg/s
        # 1: s * (Mg/s)/kN = Mg/kN
        # 2: s * (Mg/s)/kN = Mg/kN
        return [
            -(
                self.params.alpha / 2 * (gamma + prev_gamma) + self.params.mdot_bp
            ),  # d(mass)/d(dt)
            -dt * self.params.alpha / 2,  # d(mass)/d(gamma)
            -dt * self.params.alpha / 2,  # d(mass)/d(prev_gamma)
        ]

    def mass_evolution_function(self, dt, gamma, prev_gamma):
        """
        Computes the mass evolution function for the iterative step using a 1st order Taylor approximation of the original constraint.

        Args:
            dt (float): Time step duration.
            gamma (float): Current thrust magnitude.
            prev_gamma (float): Previous thrust magnitude.

        Returns:
            float: Change in mass over the time step.
        """
        previousValue = self.base_mass_evolution_function(
            dt=self.prevIterationState.dt,
            gamma=self.prevIterationState.gamma,
            prev_gamma=self.prevIterationState.prev_gamma,
        )

        derivative = self.base_mass_evolution_derivative(
            self.prevIterationState.dt,
            self.prevIterationState.gamma,
            self.prevIterationState.prev_gamma,
        )

        change = [
            dt - self.prevIterationState.dt,
            self.gamma - self.prevIterationState.gamma,
            prev_gamma - self.prevIterationState.prev_gamma,
        ]
        # 0: Mg/s * s = Mg
        # 1: Mg/kN * kN = Mg
        # 2: Mg/kN * kN = Mg
        return previousValue + self.math.dot(derivative, change)

    def base_position_evolution_derivative(self, dt, prev_vel_i, accel_i, prev_accel_i):
        """
        Computes the derivative of the base position evolution function.

        Args:
            dt (float): Time step duration.
            prev_vel_i (float): Previous velocity component.
            accel_i (float): Current acceleration component.
            prev_accel_i (float): Previous acceleration component.

        Returns:
            List[float]: Derivatives of the position evolution function with respect to dt, prev_vel_i, accel_i, and prev_accel_i.
        """
        # 0: (km/s) + (km/s^2 * s) = km/s + km/s = km/s
        # 1: s = s
        # 2: s^2 = s^2
        # 3: s^2 = s^2
        return [
            prev_vel_i + (accel_i + prev_accel_i / 2) * 2 / 3 * dt,  # d(position)/d(dt)
            dt,  # d(position)/d(prev_vel_i)
            dt**2 / 3,  # d(position)/d(accel_i)
            dt**2 / 6,  # d(position)/d(prev_accel_i)
        ]

    def position_evolution_function(self, dt, prev_vel_i, accel_i, prev_accel_i, i):
        """
        Computes the position evolution function for the iterative step using a 1st order Taylor approximation of the original constraint.

        Args:
            dt (float): Time step duration.
            prev_vel_i (float): Previous velocity component.
            accel_i (float): Current acceleration component.
            prev_accel_i (float): Previous acceleration component.
            i (int): Index of the component.

        Returns:
            float: Change in position over the time step.
        """
        previousValue = self.base_position_evolution_function(
            dt=self.prevIterationState.dt,
            prev_vel_i=self.prevIterationState.prev_velocity[i],
            accel_i=self.prevIterationState.acceleration[i],
            prev_accel_i=self.prevIterationState.prev_acceleration[i],
        )

        derivative = self.base_position_evolution_derivative(
            self.prevIterationState.dt,
            self.prevIterationState.prev_velocity[i],
            self.prevIterationState.acceleration[i],
            self.prevIterationState.prev_acceleration[i],
        )
        change = [
            dt - self.prevIterationState.dt,
            prev_vel_i - self.prevIterationState.prev_velocity[i],
            accel_i - self.prevIterationState.acceleration[i],
            prev_accel_i - self.prevIterationState.prev_acceleration[i],
        ]
        # 0: km/s * s = km
        # 1: s * km/s = km
        # 2: s^2 * km/s^2 = km
        # 3: s^2 * km/s^2 = km
        return previousValue + self.math.dot(derivative, change)

    def base_velocity_evolution_derivative(self, dt, accel_i, prev_accel_i):
        """
        Computes the derivative of the base velocity evolution function.

        Args:
            dt (float): Time step duration.
            accel_i (float): Current acceleration component.
            prev_accel_i (float): Previous acceleration component.

        Returns:
            List[float]: Derivatives of the velocity evolution function with respect to dt, accel_i, and prev_accel_i.
        """
        # 0: (km/s^2) + (km/s^2) = km/s^2
        # 1: s = s
        # 2: s = s
        return [
            (accel_i + prev_accel_i) / 2,  # d(velocity)/d(dt)
            dt / 2,  # d(velocity)/d(accel_i)
            dt / 2,  # d(velocity)/d(prev_accel_i)
        ]

    def velocity_evolution_function(self, dt, accel_i, prev_accel_i, i):
        """
        Computes the velocity evolution function for the iterative step using a 1st order Taylor approximation of the original constraint.

        Args:
            dt (float): Time step duration.
            accel_i (float): Current acceleration component.
            prev_accel_i (float): Previous acceleration component.
            i (int): Index of the component.

        Returns:
            float: Change in velocity over the time step.
        """
        previousValue = self.base_velocity_evolution_function(
            dt=self.prevIterationState.dt,
            accel_i=self.prevIterationState.acceleration[i],
            prev_accel_i=self.prevIterationState.prev_acceleration[i],
        )

        derivative = self.base_velocity_evolution_derivative(
            self.prevIterationState.dt,
            self.prevIterationState.acceleration[i],
            self.prevIterationState.prev_acceleration[i],
        )

        change = [
            dt - self.prevIterationState.dt,
            accel_i - self.prevIterationState.acceleration[i],
            prev_accel_i - self.prevIterationState.prev_acceleration[i],
        ]
        # 0: km/s^2 * s = km/s
        # 1: s * km/s^2 = km/s
        # 2: s * km/s^2 = km/s
        return previousValue + self.math.dot(derivative, change)

    def ComputeDragForce(self) -> Array3:
        """
        Computes the drag force for the iterative step. Note the linearization at play: The usage of the previous iteration's drag velocity magnitude instead of the 2-norm of the effective velocity.

        Returns:
            Array3: Drag force vector.
        """
        prev_drag_velocity = self.math.vector_add(
            self.prevIterationState.prev_velocity, self.prevIterationState.wind_velocity
        )
        prevItMag = np.linalg.norm(prev_drag_velocity)
        drag_velocity = self.math.vector_add(self.velocity, self.wind_velocity)
        self
        return self.params.ComputeDragForce(drag_velocity, prevItMag)  # kN

    def NewtonsSecondLaw(self, i):
        """
        Defines Newton's Second Law for the iterative step. Note the linearization at play: The usage of the previous iteration's mass instead of the variable mass (which would result in a bilinear term).

        Args:
            i (int): Index of the component.

        Returns:
            Constraint: Newton's Second Law constraint for the given component.
        """
        prevItMass = self.prevIterationState.mass
        # Mg * (km/s^2 + km/s^2) = kN + kN + (km/s^2 * Mg)
        # Mg * (km/s^2) = kN + (km/s^2 * Mg)
        # (kN / (kg * km/s^2)) * (Mg) * (km/s^2) = kN + (kN / (kg * km/s^2)) * (Mg) * (km/s^2)
        # kN * (Mg/kg) = kN + kN * (Mg/kg)
        # 1000 kN = kN + 1000 kN
        return (
            prevItMass * 1000 * (self.acceleration[i] + self.artificial_acceleration[i])
            == self.thrust[i]
            + self.drag_force[i]
            + self.params.g[i]
            * prevItMass
            * 1000  # TODO: Replace 2nd mass term with variable mass
        )
