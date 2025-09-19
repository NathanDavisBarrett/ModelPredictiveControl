"""
Multi_Start_Iterate_Model
=========================

This module defines the `Multi_Start_Iterate_Model` class, which extends the `Multi_Start_Base_Model` to handle multiple iterative stochastic models.
"""

from ..Stochastic import Iterate_Model, Initial_Model
from .Multi_Start_Base_Model import Multi_Start_Base_Model
from typing import List, Union

import pyomo.kernel as pmo


class Multi_Start_Iterate_Model(Multi_Start_Base_Model):
    """
    Multi_Start_Iterate_Model
    -------------------------

    Extends the `Multi_Start_Base_Model` to handle multiple iterative stochastic models.
    """

    def handle_input_models(
        self, subModels: List[Union[Iterate_Model, Initial_Model]]
    ) -> List[Iterate_Model]:
        """
        Processes the input models by converting them to `Iterate_Model` instances.

        Args:
            subModels (List[Union[Iterate_Model, Initial_Model]]): List of sub-models.

        Returns:
            List[Iterate_Model]: The processed list of iterative models.
        """
        return [Iterate_Model(previousIterationModel=model) for model in subModels]

    def handle_uniform_dt(self):
        """
        Ensures that all iterative models have the same time step by adding constraints.

        Attributes:
            uniform_dt_constraint (pmo.constraint_list): Constraints ensuring uniform time steps.
        """
        self.uniform_dt_constraint = pmo.constraint_list(
            [
                pmo.constraint(model.dt == self.subModels[0].dt)
                for model in self.subModels[1:]
            ]
        )
