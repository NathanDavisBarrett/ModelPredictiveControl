from .Initial_Parameters import (
    Initial_Parameters,
)
from ..State import IterationState

from dataclasses import dataclass
from typing import Iterable
import inspect


@dataclass(kw_only=True)
class Iterate_Parameters(Initial_Parameters):
    w_eta_dt: float = 0.1  # Weight for time step change in cost function
    w_eta_thrust: float = 10.0  # Weight for thrust change in cost function
    previous_iterate_states: Iterable[
        IterationState
    ]  # List of previous iteration states for each time step
    dt_est: float  # Estimated time step for initial guess

    @classmethod
    def from_initial_params(
        cls, initParams, initStates, dt_est, w_eta_dt=None, w_eta_thrust=None
    ):
        kwargs = {k: v for k, v in initParams.__dict__.items()}
        kwargs["dt_est"] = dt_est
        if w_eta_dt is not None:
            kwargs["w_eta_dt"] = w_eta_dt
        if w_eta_thrust is not None:
            kwargs["w_eta_thrust"] = w_eta_thrust

        kwargs["previous_iterate_states"] = initStates

        # Remove any kwargs not in the __init__ signature

        init_params = inspect.signature(cls.__init__).parameters
        valid_keys = set(init_params.keys()) - {"self"}
        kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        return cls(**kwargs)
