from ..Stochastic import Iterate_Model, Initial_Model
from .Multi_Start_Base_Model import Multi_Start_Base_Model
from typing import List, Union

import pyomo.kernel as pmo


class Multi_Start_Iterate_Model(Multi_Start_Base_Model):
    def handle_input_models(
        self, subModels: List[Union[Iterate_Model, Initial_Model]]
    ) -> List[Iterate_Model]:
        return [Iterate_Model(previousIterationModel=model) for model in subModels]

    def handle_uniform_dt(self):
        self.uniform_dt_constraint = pmo.constraint_list(
            [
                pmo.constraint(model.dt == self.subModels[0].dt)
                for model in self.subModels[1:]
            ]
        )
