"""
Initial_Model
=============

This class represents the initial model for a deterministic system. It extends the `Base_Model` class and incorporates specific initialization logic.

Attributes:
    params (Initial_Parameters): Initial parameters for the model.
    nSteps (int): Number of steps in the simulation.
    start (float): Start time of the simulation.
    stop (float): Stop time of the simulation.
    T_guess (Iterable[Array3], optional): A guess trajectory for the thrust, used as initial values.
    steps (pmo.block_list): List of step models for the simulation.

Methods:
    __init__: Initializes the initial model with the given parameters and optional guess trajectory.
"""

from .Base_Model import Base_Model
from ..Step_Model import Initial_Step_Model
from ...System import System
from ..State import InitializationState
from ..Parameters import Initial_Parameters
from ...Util.Math import Array3


import pyomo.kernel as pmo
from typing import Iterable


class Initial_Model(Base_Model):
    """
    Initial_Model
    ------------

    Represents the initial model for a deterministic system.

    Attributes:
        params (Initial_Parameters): Initial parameters for the model.
        nSteps (int): Number of steps in the simulation.
        start (float): Start time of the simulation.
        stop (float): Stop time of the simulation.
        T_guess (Iterable[Array3], optional): A guess trajectory for the thrust, used as initial values.
        steps (pmo.block_list): List of step models for the simulation.
    """

    def __init__(
        self,
        params: Initial_Parameters,
        nSteps: int,
        start: float,
        stop: float,
        T_guess: Iterable[Array3] = None,
    ):
        """
        Initializes the initial model with the given parameters and optional guess trajectory.

        Args:
            params (Initial_Parameters): Initial parameters for the model.
            nSteps (int): Number of steps in the simulation.
            start (float): Start time of the simulation.
            stop (float): Stop time of the simulation.
            T_guess (Iterable[Array3], optional): A guess trajectory for the thrust, used as initial values.
        """
        assert stop is not None, "Stop time must be provided for initial model"
        super().__init__(params, nSteps, start, stop)

        initialSpeed = params.math.norm(params.v0)
        finalSpeed = params.math.norm(params.vf)
        dsdt = (finalSpeed - initialSpeed) / (stop - start)
        dmdt = (params.m_dry - params.m0) / (stop - start)

        if T_guess is not None:
            # A guess trajectory is provided, load it into the model as the initial values.
            system = System(
                params, reference_mass=params.m0, reference_speed=initialSpeed
            )

        self.steps = pmo.block_list()
        for i in range(nSteps):
            isFinal = i == nSteps - 1
            prevState = self.initState if i == 0 else self.steps[i - 1].getState()

            reference_mass = params.m0 + dmdt * (i * self.dt)
            reference_speed = initialSpeed + dsdt * (i * self.dt)

            if T_guess is not None:
                system.reference_mass = reference_mass
                system.reference_speed = reference_speed

                system.step(T_guess[i], self.dt)

                initializationState = InitializationState.from_system(system)
            else:
                initializationState = None

            step = Initial_Step_Model(
                t_est=i * self.dt,
                params=params,
                dt=self.dt,
                reference_mass=reference_mass,
                reference_speed=reference_speed,
                prevState=prevState,
                isFinal=isFinal,
                initializationState=initializationState,
            )
            self.steps.append(step)

        self.artificial_acceleration_norm = pmo.expression(
            sum([step.artificial_acceleration_mag**2 for step in self.steps])
        )  # DEPARTING FROM THE ORIGINAL PAPER HERE. Originally, the 2-norm, but the sum of squares is better to penalize large spikes and allows the gurobi solver so solve the model very quickly (since the objective is now quadratic).

        self.objective_expr = pmo.expression(
            -params.w_m * self.steps[-1].mass
            + params.w_a * self.artificial_acceleration_norm
        )

        self.objective = pmo.objective(
            expr=self.objective_expr,
            sense=pmo.minimize,
        )
