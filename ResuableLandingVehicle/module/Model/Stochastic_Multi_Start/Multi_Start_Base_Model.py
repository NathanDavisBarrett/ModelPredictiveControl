"""
Multi_Start_Base_Model
======================

This module defines the `Multi_Start_Base_Model` class, which serves as an abstract base class for multi-start stochastic models.
"""

from ..Stochastic import Iterate_Model, Initial_Model
from typing import List, Union

import pyomo.kernel as pmo
from itertools import chain
from abc import ABC, abstractmethod


class Multi_Start_Base_Model(pmo.block, ABC):
    """
    Multi_Start_Base_Model
    -----------------------

    Abstract base class for multi-start stochastic models.

    Attributes:
        subModels (pmo.block_list): List of sub-models (either `Iterate_Model` or `Initial_Model`).
        objective (pmo.objective): Objective function for the multi-start model.
        non_anticipative_thrust (pmo.constraint_dict): Constraints ensuring non-anticipative thrust.
    """

    def __init__(
        self,
        subModels: List[Union[Iterate_Model, Initial_Model]],
        num_non_anticipative_periods: int,
    ):
        """
        Initializes the `Multi_Start_Base_Model` with the given sub-models and non-anticipative periods.

        Args:
            subModels (List[Union[Iterate_Model, Initial_Model]]): List of sub-models.
            num_non_anticipative_periods (int): Number of non-anticipative periods.
        """
        super().__init__()

        self.subModels = pmo.block_list(self.handle_input_models(subModels))

        # Deactivate all the objective functions in the deterministic models
        for model in self.subModels:
            model.objective.deactivate()

        num_scenarios = len(self.subModels)

        self.objective = pmo.objective(
            expr=sum(model.objective_expr / num_scenarios for model in self.subModels),
            sense=pmo.minimize,
        )

        self.non_anticipative_thrust = pmo.constraint_dict()
        for time_index in range(num_non_anticipative_periods):
            allSteps = []
            for model_index in range(num_scenarios):
                allSteps.append(
                    self.subModels[model_index].iter_nodes_at_depth(time_index)
                )
            allSteps = chain(*allSteps)
            base_step = next(allSteps)
            for i, step in enumerate(allSteps):
                for dim in range(3):
                    self.non_anticipative_thrust[time_index, i, dim] = pmo.constraint(
                        base_step.thrust[dim] == step.thrust[dim]
                    )

        # Initial models must all have the same time step
        self.handle_uniform_dt()

    @abstractmethod
    def handle_input_models(
        self, subModels: List[Union[Iterate_Model, Initial_Model]]
    ) -> List[Union[Iterate_Model, Initial_Model]]:
        """
        Abstract method to handle input models.

        Args:
            subModels (List[Union[Iterate_Model, Initial_Model]]): List of sub-models.

        Returns:
            List[Union[Iterate_Model, Initial_Model]]: Processed list of sub-models.
        """
        pass

    @abstractmethod
    def handle_uniform_dt(self):
        """
        Abstract method to ensure uniform time steps across models.
        """
        pass

    def Plot(self, axDict: dict = None, saveFileName: str = None):
        """
        Plots the results of the multi-start model.

        Args:
            axDict (dict, optional): Dictionary of axes for plotting. Defaults to None.
            saveFileName (str, optional): File name to save the animation. Defaults to None.

        Returns:
            dict: Dictionary of axes used for plotting.
        """
        if axDict is None:
            import matplotlib.pyplot as plt

            show = True
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(3, 2)
            posAx = fig.add_subplot(gs[0, 0])
            velAx = fig.add_subplot(gs[0, 1])
            accAx = fig.add_subplot(gs[1, 0])
            massAx = fig.add_subplot(gs[1, 1])
            thrustAx = fig.add_subplot(gs[2, 0])

            threeDimAx = fig.add_subplot(gs[2, 1], projection="3d")

            axDict = {
                "posAx": posAx,
                "velAx": velAx,
                "accAx": accAx,
                "massAx": massAx,
                "thrustAx": thrustAx,
                "threeDimAx": threeDimAx,
            }
        else:
            show = False

        for i, model in enumerate(self.subModels):
            model.Plot(axDict, withLabels=(i == 0))

        if saveFileName is not None:
            threeDimAx = axDict["threeDimAx"]
            import matplotlib.animation as animation

            def rotate(angle):
                threeDimAx.view_init(elev=30, azim=angle)

            ani = animation.FuncAnimation(
                fig=threeDimAx.figure,
                func=rotate,
                frames=range(0, 360, 2),
                interval=50,
                blit=False,
            )

            ani.save(saveFileName, writer="pillow")

        if show:
            plt.show()

        return axDict
