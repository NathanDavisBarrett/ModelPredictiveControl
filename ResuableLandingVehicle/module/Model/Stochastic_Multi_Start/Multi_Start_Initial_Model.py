"""
Multi_Start_Initial_Model
=========================

This module defines the `Multi_Start_Initial_Model` class, which extends the `Multi_Start_Base_Model` to handle multiple initial stochastic models.
"""

from ..Stochastic import Initial_Model
from .Multi_Start_Base_Model import Multi_Start_Base_Model
from typing import List


class Multi_Start_Initial_Model(Multi_Start_Base_Model):
    """
    Multi_Start_Initial_Model
    -------------------------

    Extends the `Multi_Start_Base_Model` to handle multiple initial stochastic models.
    """

    def handle_input_models(
        self, subModels: List[Initial_Model]
    ) -> List[Initial_Model]:
        """
        Processes the input models.

        Args:
            subModels (List[Initial_Model]): List of initial stochastic models.

        Returns:
            List[Initial_Model]: The processed list of initial models.
        """
        return subModels

    def handle_uniform_dt(self):
        """
        Ensures that all initial models have the same time step.

        Raises:
            AssertionError: If the time steps of the models are not uniform.
        """
        for model_index in range(1, len(self.subModels)):
            assert (
                self.subModels[0].dt == self.subModels[model_index].dt
            ), "All initial models must have the same time step"
