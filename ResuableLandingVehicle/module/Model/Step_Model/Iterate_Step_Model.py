from .Base_Step_Model import Base_Step_Model
from ...System import SystemParameters
from ...Util.Math import Array3
from ..State import State, InitializationState, IterationState

import pyomo.kernel as pmo
import numpy as np
from typing import Union


class Iterate_Step_Model(Base_Step_Model):
    def __init__(
        self,
        t_est: float,  # Estimated time (used for determining wind speed)
        params: SystemParameters,
        dt: Union[float, pmo.variable],
        prevTimeState: State,
        prevIterationState: IterationState,
        isFinal: bool = False,
    ):

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
        super().variables(initializationState=initializationState)

        thrust_change = self.math.vector_add(
            initializationState.thrust,
            self.math.vector_scale(self.prevIterationState.prev_thrust, -1),
        )

        eta_val = self.math.dot(thrust_change, thrust_change)

        self.eta_thrust = pmo.variable(domain=pmo.NonNegativeReals, value=eta_val)

    def constraints(self):
        super().constraints()

        # Nonlinear, Convex Constraint
        thrust_change = self.math.vector_add(
            self.thrust, self.math.vector_scale(self.prevIterationState.prev_thrust, -1)
        )
        self.eta_thrust_def = pmo.constraint(
            self.math.dot(thrust_change, thrust_change) <= self.eta_thrust
        )

    def base_mass_evolution_derivative(self, dt, gamma, prev_gamma):
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
        # 0: (km/s^2) + (km/s^2) = km/s^2
        # 1: s = s
        # 2: s = s
        return [
            (accel_i + prev_accel_i) / 2,  # d(velocity)/d(dt)
            dt / 2,  # d(velocity)/d(accel_i)
            dt / 2,  # d(velocity)/d(prev_accel_i)
        ]

    def velocity_evolution_function(self, dt, accel_i, prev_accel_i, i):
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
        prev_drag_velocity = self.math.vector_add(
            self.prevIterationState.prev_velocity, self.prevIterationState.wind_velocity
        )
        prevItMag = np.linalg.norm(prev_drag_velocity)
        drag_velocity = self.math.vector_add(self.velocity, self.wind_velocity)
        self
        return self.params.ComputeDragForce(drag_velocity, prevItMag)  # kN

    def NewtonsSecondLaw(self, i):
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
