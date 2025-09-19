# ReusableLandingVehicle

![](MultiStart_Iteration.gif)

## Motivation
Reusable landing vehicles represent a transformative innovation in the space industry, offering the potential to dramatically reduce the cost of access to space. By enabling rockets and spacecraft to return safely to Earth and be launched again, these technologies promote sustainability and efficiency, making frequent missions more economically viable. This capability not only accelerates the pace of exploration and satellite deployment but also opens new possibilities for commercial ventures and scientific research, driving progress toward a more accessible and sustainable future in space.

## Key Features

* **Sequential Convexification (Optimization Theory and Practice)**
> The dynamics of a fuel-optimal landing of a space vehicle are very non-linear and non-convex, especially when considering <u>variable time step durations</u>. Thus, determining optimal (or even feasible) solutions can be remarkably challenging. 
> 
> However, with some clever modifications, the non-convexities present in the model can be relaxed to produce a convex (and thus easily and <u>quickly solvable</u>) model. These convexifications involve a 1st-order Taylor approximation of the most grievous non-convexities that are updated at each iteration to provide a more locally-accurate representation. With some personal adaptations, I was able to further re-formulate the original problem as a <u>Quadratic</u> Second-Order Cone Program (a sub-class of Convex Quadratically-Constrained Quadratic Programs QCQP) for which even large problem instances solved to global optimality in a fraction of a second.

* **General Stochastic Programming Architecture (Estimation and Uncertainty Modeling)**
> The major contribution that I personally did (beyond the paper cited below) is the introduction of a general stochastic programming architecture. Thus, my improved model can handle large degrees of <u>uncertainty</u> via a series of Monte Carlo-generated scenarios that can be connected in any general fashion (e.g. linear, branching, etc.).
>
> Primarily, I considered uncertainty in the following parameters (though it is trivial to consider more):
> * Wind Speed
> * Initial Position
> * Initial Velocity
> * Initial Mass
> * Specific Impulse
> * Air Pressure
> * Air Density

* **Rigorous Software Engineering Practices**
> Throughout my implementation, I've adopted several best practices pertaining to the software engineering of this project. Some of the most notable are as follows:
> * **Thorough Documentation** of every function and class, leveraging Python Type Hints to ensure clarity and robustness.
> * **Object-Oriented Design**: Every part of this project relies heavily on object-oriented programming principles (inheritance, polymorphism, encapsulation, etc.) to ensure scalability and robustness.
> * **Single Responsibility Principle**: Each class, function, file, and module serves a single purpose. Doing so drastically improves the <u>maintainability and adaptability</u> of the code. This is evident in that the core model "Base_Step_Model" drives every part of the initial, iterated, deterministic, and stochastic models.
> * **Numerous Unit Tests**: Each modeling component is tested thoroughly before being utilized in other parts of the module.


## Citation
This entire project, including its ideological approach, variables, equations, etc., is inspired heavily by the following article:

> M. Szmuk, B. Acikmese, and A. W. Berning, "Successive Convexification for Fuel-Optimal Powered Landing with Aerodynamic Drag and Non-Convex Constraints," in *AIAA Guidance, Navigation, and Control Conference*. doi: 10.2514/6.2016-0378.

Beyond some reformulations (particularly of the objective), my major contribution was introducing a stochastic element to the mathematical framework.