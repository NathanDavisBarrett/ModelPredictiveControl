from ..Stochastic import Initial_Model
from .Multi_Start_Base_Model import Multi_Start_Base_Model
from typing import List


class Multi_Start_Initial_Model(Multi_Start_Base_Model):
    def handle_input_models(
        self, subModels: List[Initial_Model]
    ) -> List[Initial_Model]:
        return subModels

    def handle_uniform_dt(self):
        for model_index in range(1, len(self.subModels)):
            assert (
                self.subModels[0].dt == self.subModels[model_index].dt
            ), "All initial models must have the same time step"
