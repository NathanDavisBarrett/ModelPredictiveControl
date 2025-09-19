"""
Iterate_Model
=============

This class represents the iterative model for a deterministic system. It extends the `Base_Model` class and incorporates logic for iterative optimization.

Attributes:
    params (Iterate_Parameters): Parameters for the iterative model.
    nSteps (int): Number of steps in the simulation.
    start (float): Start time of the simulation.
    steps (pmo.block_list): List of step models for the simulation.
    artificial_acceleration_norm (pmo.expression): Expression to penalize large spikes in artificial acceleration.
    thrust_change_norm (pmo.expression): Expression to penalize large spikes in thrust changes.
    eta_dt (pmo.variable): Variable for time step adjustment.

Methods:
    __init__: Initializes the iterative model with the given parameters.
"""

from .Base_Model import Base_Model
from ..Parameters import Iterate_Parameters
from ..Step_Model import Iterate_Step_Model

import pyomo.kernel as pmo


class Iterate_Model(Base_Model):
    """
    Iterate_Model
    ------------

    Represents the iterative model for a deterministic system.

    Attributes:
        params (Iterate_Parameters): Parameters for the iterative model.
        nSteps (int): Number of steps in the simulation.
        start (float): Start time of the simulation.
        steps (pmo.block_list): List of step models for the simulation.
        artificial_acceleration_norm (pmo.expression): Expression to penalize large spikes in artificial acceleration.
        thrust_change_norm (pmo.expression): Expression to penalize large spikes in thrust changes.
        eta_dt (pmo.variable): Variable for time step adjustment.
    """

    def __init__(
        self,
        params: Iterate_Parameters,
        nSteps: int,
        start: float,
    ):
        """
        Initializes the iterative model with the given parameters.

        Args:
            params (Iterate_Parameters): Parameters for the iterative model.
            nSteps (int): Number of steps in the simulation.
            start (float): Start time of the simulation.
        """
        super().__init__(params, nSteps, start, stop=None)

        self.steps = pmo.block_list()
        for i in range(nSteps):
            isFinal = i == nSteps - 1
            prevState = self.initState if i == 0 else self.steps[i - 1].getState()

            step = Iterate_Step_Model(
                t_est=i * self.params.dt_est,
                params=params,
                dt=self.dt,
                prevTimeState=prevState,
                prevIterationState=params.previous_iterate_states[i],
                isFinal=isFinal,
            )
            self.steps.append(step)

        self.artificial_acceleration_norm = pmo.expression(
            sum([step.artificial_acceleration_mag**2 for step in self.steps])
        )  # DEPARTING FROM THE ORIGINAL PAPER HERE. Originally, the 2-norm, but the sum of squares is better to penalize large spikes.

        self.thrust_change_norm = pmo.expression(
            sum([step.eta_thrust**2 for step in self.steps])
        )  # DEPARTING FROM THE ORIGINAL PAPER HERE. Originally, the 2-norm, but the sum of squares is better to penalize large spikes.

        self.eta_dt = pmo.variable(domain=pmo.NonNegativeReals, value=1.0)
        dt_change = self.dt - params.previous_iterate_states[0].dt
        self.eta_dt_def = pmo.constraint(dt_change**2 <= self.eta_dt)

        self.objective_expr = pmo.expression(
            -params.w_m * self.steps[-1].mass
            + params.w_eta_dt * self.eta_dt
            + params.w_eta_thrust * self.thrust_change_norm
            + params.w_a * self.artificial_acceleration_norm
        )

        self.objective = pmo.objective(
            expr=self.objective_expr,
            sense=pmo.minimize,
        )
