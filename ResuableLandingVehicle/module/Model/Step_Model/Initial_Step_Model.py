"""
Initial_Step_Model
==================

This module defines the `Initial_Step_Model` class, which extends the `Base_Step_Model` to provide specific implementations for the initial step in a simulation.
"""

from .Base_Step_Model import Base_Step_Model
from ...System import SystemParameters
from ...Util.Math import Array3
from ..State import State, InitializationState


class Initial_Step_Model(Base_Step_Model):
    """
    Initial_Step_Model
    ------------------

    Extends the `Base_Step_Model` to provide specific implementations for the initial step in a simulation.

    Attributes:
        t_est (float): Estimated time for determining wind speed.
        params (SystemParameters): System parameters for the simulation.
        dt (float): Fixed time step duration for the initial guess.
        reference_mass (float): Reference mass for the initial step.
        reference_speed (Array3): Reference speed for the initial step.
        prevState (State): State of the system in the previous step.
        isFinal (bool): Indicates whether this is the final step.
        initializationState (InitializationState): Initial state of the system.
    """

    def __init__(
        self,
        t_est: float,
        params: SystemParameters,
        dt: float,
        reference_mass: float = None,
        reference_speed: Array3 = None,
        prevState: State = None,
        isFinal: bool = False,
        initializationState: InitializationState = None,
    ):
        """
        Initializes the `Initial_Step_Model` with the given parameters.

        Args:
            t_est (float): Estimated time for determining wind speed.
            params (SystemParameters): System parameters for the simulation.
            dt (float): Fixed time step duration for the initial guess.
            reference_mass (float, optional): Reference mass for the initial step. Defaults to None.
            reference_speed (Array3, optional): Reference speed for the initial step. Defaults to None.
            prevState (State, optional): State of the system in the previous step. Defaults to None.
            isFinal (bool, optional): Indicates whether this is the final step. Defaults to False.
            initializationState (InitializationState, optional): Initial state of the system. Defaults to None.
        """

        super().__init__(
            t_est=t_est,
            params=params,
            dt=dt,
            prevState=prevState,
            isFinal=isFinal,
            otherParams=dict(
                reference_mass=reference_mass, reference_speed=reference_speed
            ),
            initializationState=initializationState,
        )

    def mass_evolution_function(self, dt, gamma, prev_gamma):
        """
        Computes the mass evolution function for the initial step.

        Args:
            dt (float): Time step duration.
            gamma (float): Current thrust magnitude.
            prev_gamma (float): Previous thrust magnitude.

        Returns:
            float: Change in mass over the time step.
        """
        return self.base_mass_evolution_function(dt, gamma, prev_gamma)

    def position_evolution_function(self, dt, prev_vel_i, accel_i, prev_accel_i, _):
        """
        Computes the position evolution function for the initial step.

        Args:
            dt (float): Time step duration.
            prev_vel_i (float): Previous velocity component.
            accel_i (float): Current acceleration component.
            prev_accel_i (float): Previous acceleration component.

        Returns:
            float: Change in position over the time step.
        """
        return self.base_position_evolution_function(
            dt, prev_vel_i, accel_i, prev_accel_i
        )

    def velocity_evolution_function(self, dt, accel_i, prev_accel_i, _):
        """
        Computes the velocity evolution function for the initial step.

        Args:
            dt (float): Time step duration.
            accel_i (float): Current acceleration component.
            prev_accel_i (float): Previous acceleration component.

        Returns:
            float: Change in velocity over the time step.
        """
        return self.base_velocity_evolution_function(dt, accel_i, prev_accel_i)

    def ComputeDragForce(self) -> Array3:
        """
        Computes the drag force for the initial step. Note the linearization at play: The usage of reference speed instead of the 2-norm of the effective velocity.

        Returns:
            Array3: Drag force vector.
        """
        v_eff = self.math.vector_add(self.velocity, self.wind_velocity)
        return self.params.ComputeDragForce(v_eff, v_mag=self.reference_speed)

    def NewtonsSecondLaw(self, i):
        """
        Defines Newton's Second Law for the initial step. Note the linearization at play: The usage of reference mass instead of the variable mass (which would result in a bilinear term).

        Args:
            i (int): Index of the component.

        Returns:
            Constraint: Newton's Second Law constraint for the given component.
        """
        return (
            self.reference_mass
            * 1000
            * (self.acceleration[i] + self.artificial_acceleration[i])
            == self.thrust[i]
            + self.drag_force[i]
            + self.params.g[i]
            * self.reference_mass
            * 1000  # TODO: Replace 2nd mass term with variable mass
        )
