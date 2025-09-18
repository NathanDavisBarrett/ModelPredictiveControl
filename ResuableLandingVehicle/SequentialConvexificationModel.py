from System import SystemParameters, Array3, System

from typing import Union, Iterable, Dict, Any

import pyomo.kernel as pmo
import numpy as np
from PyomoTools.kernel.Formulations import Conic
from dataclasses import dataclass
from abc import ABC, abstractmethod


class State:
    def __init__(
        self,
        mass: Union[float, pmo.variable],
        position: Array3,
        velocity: Array3,
        acceleration: Array3,
        thrust: Array3,
        gamma: Union[float, pmo.variable],
    ):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.thrust = thrust
        self.gamma = gamma


class InitializationState:
    def __init__(
        self,
        mass: float,
        position: Array3,
        velocity: Array3,
        acceleration: Array3,
        artificial_acceleration: Array3,
        thrust: Array3,
    ):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.artificial_acceleration = artificial_acceleration
        self.thrust = thrust

    @property
    def gamma(self) -> float:
        return np.linalg.norm(self.thrust)

    @property
    def artificial_acceleration_mag(self) -> float:
        return np.linalg.norm(self.artificial_acceleration)

    @classmethod
    def from_system(cls, system: System):
        return cls(
            mass=system.m,
            position=system.x,
            velocity=system.v,
            acceleration=[0.0, 0.0, 0.0],
            artificial_acceleration=[0.0, 0.0, 0.0],
            thrust=system.T,
        )


class IterationState(InitializationState):
    def __init__(
        self,
        dt: float,
        mass: float,
        prev_mass: float,
        gamma: float,
        prev_gamma: float,
        velocity: Array3,
        prev_velocity: Array3,
        thrust: Array3,
        prev_thrust: Array3,
        acceleration: Array3,
        prev_acceleration: Array3,
        position: Array3,
        artificial_acceleration: Array3,
    ):
        super().__init__(
            mass=mass,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            artificial_acceleration=artificial_acceleration,
            thrust=thrust,
        )
        self.dt = dt
        self.prev_mass = prev_mass
        self._gamma = gamma
        self.prev_gamma = prev_gamma
        self.prev_velocity = prev_velocity
        self.prev_thrust = prev_thrust
        self.prev_acceleration = prev_acceleration

        @property
        def gamma(self) -> float:
            return self._gamma


@dataclass
class SequentialConvexification_Initial_Parameters(SystemParameters):
    w_m: float = 1.0  # Weight for mass in cost function
    w_a: float = 100.0  # Weight for artificial acceleration in cost function


@dataclass(kw_only=True)
class SequentialConvexification_Iterate_Parameters(
    SequentialConvexification_Initial_Parameters
):
    w_eta_dt: float = 10.0  # Weight for time step change in cost function
    w_eta_thrust: float = 10.0  # Weight for thrust change in cost function
    previous_iterate_states: Iterable[
        IterationState
    ]  # List of previous iteration states for each time step

    @classmethod
    def from_initial_params(
        cls, initParams, initStates, w_eta_dt=None, w_eta_thrust=None
    ):
        kwargs = {k: v for k, v in initParams.__dict__.items()}
        if w_eta_dt is not None:
            kwargs["w_eta_dt"] = w_eta_dt
        if w_eta_thrust is not None:
            kwargs["w_eta_thrust"] = w_eta_thrust

        kwargs["previous_iterate_states"] = initStates

        return cls(**kwargs)


class SequentialConvexification_Base_Step_Model(pmo.block, ABC):
    X = 0
    Y = 1
    Z = 2

    def __init__(
        self,
        params: SystemParameters,
        dt: Union[float, pmo.variable],
        prevState: State = None,
        isFinal: bool = False,
        otherParams: Dict[str, Any] = {},
        initializationState: InitializationState = None,
    ):
        super().__init__()

        self.params = params
        for key, value in otherParams.items():
            setattr(self, key, value)
        self._dt = (
            lambda: dt
        )  # Keeping this as a function allows for referencing without this step model trying to take ownership of the variable.
        self.math = params.math
        self.prevState = prevState

        self.variables(initializationState=initializationState)
        self.constraints()
        if isFinal:
            self.final_constraints()

    @property
    def dt(self):
        return self._dt()

    def variables(self, initializationState: InitializationState = None):

        mass_val = (
            pmo.value(self.prevState.mass)
            if initializationState is None
            else initializationState.mass
        )
        self.mass = pmo.variable(lb=self.params.m_dry, value=mass_val)

        poss_val = (
            [pmo.value(self.prevState.position[i]) for i in range(3)]
            if initializationState is None
            else initializationState.position
        )
        self.position = pmo.variable_list(
            [pmo.variable(value=poss_val[i]) for i in range(3)]
        )

        vel_val = (
            [pmo.value(self.prevState.velocity[i]) for i in range(3)]
            if initializationState is None
            else initializationState.velocity
        )
        if np.allclose(vel_val, [0.0, 0.0, 0.0]):
            vel_val = [1e-5, 1e-5, 1e-5]  # Avoid numerical issues with zero velocity
        self.velocity = pmo.variable_list(
            [pmo.variable(value=vel_val[i]) for i in range(3)]
        )

        accel_val = (
            [pmo.value(self.prevState.acceleration[i]) for i in range(3)]
            if initializationState is None
            else initializationState.acceleration
        )
        self.acceleration = pmo.variable_list(
            [pmo.variable(value=accel_val[i]) for i in range(3)]
        )

        art_accel_val = (
            [0.0, 0.0, 0.0]
            if initializationState is None
            else initializationState.artificial_acceleration
        )
        self.artificial_acceleration = pmo.variable_list(
            [pmo.variable(value=art_accel_val[i]) for i in range(3)]
        )

        art_accel_mag_val = (
            1e-5
            if initializationState is None
            else initializationState.artificial_acceleration_mag
        )
        self.artificial_acceleration_mag = pmo.variable(
            domain=pmo.NonNegativeReals, value=art_accel_mag_val
        )

        thrust_val = (
            [pmo.value(self.prevState.thrust[i]) for i in range(3)]
            if initializationState is None
            else initializationState.thrust
        )
        self.thrust = pmo.variable_list(
            [pmo.variable(value=thrust_val[i]) for i in range(3)]
        )

        gamma_val = (
            pmo.value(self.prevState.gamma)
            if initializationState is None
            else initializationState.gamma
        )
        self.gamma = pmo.variable(
            domain=pmo.NonNegativeReals,
            value=gamma_val,
        )

        df = self.ComputeDragForce()

        self.drag_force = pmo.variable_list(
            [pmo.variable(value=pmo.value(df[i])) for i in range(3)]
        )

    def base_mass_evolution_function(self, dt, gamma, prev_gamma):
        return -dt * (
            self.params.alpha / 2 * (gamma + prev_gamma) + self.params.mdot_bp
        )

    @abstractmethod
    def mass_evolution_function(self, dt, gamma, prev_gamma):
        raise NotImplementedError(
            "Subclasses must implement mass_evolution_function method"
        )

    def base_position_evolution_function(self, dt, prev_vel_i, accel_i, prev_accel_i):
        return dt * prev_vel_i + (accel_i + prev_accel_i / 2) / 3 * dt**2

    @abstractmethod
    def position_evolution_function(self, dt, prev_vel_i, accel_i, prev_accel_i):
        raise NotImplementedError(
            "Subclasses must implement position_evolution_function method"
        )

    def base_velocity_evolution_function(self, dt, accel_i, prev_accel_i):
        return (accel_i + prev_accel_i) / 2 * dt

    @abstractmethod
    def velocity_evolution_function(self, dt, accel_i, prev_accel_i):
        raise NotImplementedError(
            "Subclasses must implement velocity_evolution_function method"
        )

    @abstractmethod
    def ComputeDragForce(self) -> Array3:
        raise NotImplementedError("Subclasses must implement ComputeDragForce method")

    @abstractmethod
    def NewtonsSecondLaw(self, i):
        raise NotImplementedError("Subclasses must implement NewtonsSecondLaw method")

    def constraints(self):
        # Linear Constraint (constant dt)
        self.mass_evolution = pmo.constraint(
            self.mass - self.prevState.mass
            == self.mass_evolution_function(self.dt, self.gamma, self.prevState.gamma)
        )

        self.position_evolution = pmo.constraint_list()
        self.velocity_evolution = pmo.constraint_list()
        self.newtons_second_law = pmo.constraint_list()
        self.drag_force_def = pmo.constraint_list()

        df = self.ComputeDragForce()

        for i in range(3):
            # Linear Constraint (constant dt)
            self.position_evolution.append(
                pmo.constraint(
                    self.position[i] - self.prevState.position[i]
                    == self.position_evolution_function(
                        self.dt,
                        self.prevState.velocity[i],
                        self.acceleration[i],
                        self.prevState.acceleration[i],
                        i,
                    )
                )
            )

            # Linear Constraint (constant dt)
            self.velocity_evolution.append(
                pmo.constraint(
                    self.velocity[i] - self.prevState.velocity[i]
                    == self.velocity_evolution_function(
                        self.dt, self.acceleration[i], self.prevState.acceleration[i], i
                    )
                )
            )

            # Linear Constraint
            self.newtons_second_law.append(pmo.constraint(self.NewtonsSecondLaw(i)))

            # Linear Constraint
            self.drag_force_def.append(pmo.constraint(self.drag_force[i] == df[i]))

        # Nonlinear, Convex Constraint
        self.artificial_acceleration_mag_def = Conic(
            r=self.artificial_acceleration_mag,
            x=self.artificial_acceleration,
            order=2,
        )

        # Nonlinear, Convex Constraint
        self.gamma_def = Conic(r=self.gamma, x=self.thrust, order=2)

        # Nonlinear, Convex Constraint
        relative_position = self.math.vector_add(
            self.position, self.math.vector_scale(self.params.xf, -1)
        )
        self.glide_slope = Conic(
            r=self.math.dot(self.params.e_u, relative_position)
            / np.cos(self.params.max_glide_slope),
            x=relative_position,
            order=2,
        )

        # Linear Constraint
        self.thrust_lower_limit = pmo.constraint(self.gamma >= self.params.T_min)

        # Linear Constraint
        self.thrust_upper_limit = pmo.constraint(self.gamma <= self.params.T_max)

        # Linear Constraint
        self.tilt_angle = pmo.constraint(
            self.math.dot(self.params.e_u, self.thrust)
            >= np.cos(self.params.max_tilt) * self.gamma
        )

        # Linear Constraint
        self.trust_ramp_upper_limit = pmo.constraint(
            (self.gamma - self.prevState.gamma) <= self.params.dTdt_max * self.dt
        )

        # Linear Constraint
        self.trust_ramp_lower_limit = pmo.constraint(
            (self.gamma - self.prevState.gamma) >= self.params.dTdt_min * self.dt
        )

    def final_constraints(self):
        self.final_position = pmo.constraint_list()
        self.final_velocity = pmo.constraint_list()
        self.final_direction = pmo.constraint_list()
        for i in range(3):
            self.final_position.append(
                pmo.constraint(self.position[i] == self.params.xf[i])
            )
            self.final_velocity.append(
                pmo.constraint(self.velocity[i] == self.params.vf[i])
            )
            self.final_direction.append(
                pmo.constraint(self.thrust[i] == self.gamma * self.params.nf[i])
            )

    def getState(self) -> State:
        return State(
            mass=self.mass,
            position=self.position,
            velocity=self.velocity,
            acceleration=self.acceleration,
            thrust=self.thrust,
            gamma=self.gamma,
        )

    def getIterationState(self) -> IterationState:
        return IterationState(
            dt=pmo.value(self.dt),
            mass=pmo.value(self.mass),
            prev_mass=pmo.value(self.prevState.mass),
            gamma=pmo.value(self.gamma),
            prev_gamma=pmo.value(self.prevState.gamma),
            velocity=[pmo.value(v) for v in self.velocity],
            prev_velocity=[pmo.value(v) for v in self.prevState.velocity],
            thrust=[pmo.value(t) for t in self.thrust],
            prev_thrust=[pmo.value(t) for t in self.prevState.thrust],
            acceleration=[pmo.value(a) for a in self.acceleration],
            prev_acceleration=[pmo.value(a) for a in self.prevState.acceleration],
            position=[pmo.value(p) for p in self.position],
            artificial_acceleration=[
                pmo.value(a) for a in self.artificial_acceleration
            ],
        )


class SequentialConvexification_Initial_Step_Model(
    SequentialConvexification_Base_Step_Model
):
    def __init__(
        self,
        params: SystemParameters,
        dt: float,  # Float must be fixed for the initial guess
        reference_mass: float = None,
        reference_speed: Array3 = None,
        prevState: State = None,
        isFinal: bool = False,
        initializationState: InitializationState = None,
    ):

        super().__init__(
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
        return self.base_mass_evolution_function(dt, gamma, prev_gamma)

    def position_evolution_function(self, dt, prev_vel_i, accel_i, prev_accel_i, _):
        return self.base_position_evolution_function(
            dt, prev_vel_i, accel_i, prev_accel_i
        )

    def velocity_evolution_function(self, dt, accel_i, prev_accel_i, _):
        return self.base_velocity_evolution_function(dt, accel_i, prev_accel_i)

    def ComputeDragForce(self) -> Array3:
        return self.params.ComputeDragForce(self.velocity, v_mag=self.reference_speed)

    def NewtonsSecondLaw(self, i):
        return (
            self.reference_mass
            * (self.acceleration[i] + self.artificial_acceleration[i])
            == self.thrust[i]
            + self.drag_force[i]
            + self.params.g[i] * self.reference_mass  # self. mass
        )  # NOTE: DEPARTING FROM THE ORIGINAL PAPER HERE. Originally, reference_mass was used instead in both locations. But regular mass can be used here without introducing non-convexity.


class SequentialConvexification_Iterate_Step_Model(
    SequentialConvexification_Base_Step_Model
):
    def __init__(
        self,
        params: SystemParameters,
        dt: Union[float, pmo.variable],
        prevTimeState: State,
        prevIterationState: IterationState,
        isFinal: bool = False,
    ):

        super().__init__(
            params=params,
            dt=dt,
            prevState=prevTimeState,
            isFinal=isFinal,
            otherParams=dict(
                prevTimeState=prevTimeState, prevIterationState=prevIterationState
            ),
            initializationState=prevIterationState,
        )

    def variables(self, initializationState: InitializationState = None):
        super().variables(initializationState=initializationState)

        thrust_change = self.math.vector_add(
            initializationState.thrust,
            self.math.vector_scale(self.prevIterationState.prev_thrust, -1),
        )

        eta_val = self.math.dot(thrust_change, thrust_change)

        self.eta_thrust = pmo.variable(domain=pmo.NonNegativeReals, value=eta_val)

    def constraints(self):
        super().constraints()

        # Nonlinear, Convex Constraint
        thrust_change = self.math.vector_add(
            self.thrust, self.math.vector_scale(self.prevIterationState.prev_thrust, -1)
        )
        self.eta_thrust_def = pmo.constraint(
            self.math.dot(thrust_change, thrust_change) <= self.eta_thrust
        )

    def base_mass_evolution_derivative(self, dt, gamma, prev_gamma):
        return [
            -(
                self.params.alpha / 2 * (gamma + prev_gamma) + self.params.mdot_bp
            ),  # d(mass)/d(dt)
            -dt * self.params.alpha / 2,  # d(mass)/d(gamma)
            -dt * self.params.alpha / 2,  # d(mass)/d(prev_gamma)
        ]

    def mass_evolution_function(self, dt, gamma, prev_gamma):
        previousValue = self.base_mass_evolution_function(
            dt=self.prevIterationState.dt,
            gamma=self.prevIterationState.gamma,
            prev_gamma=self.prevIterationState.prev_gamma,
        )

        derivative = self.base_mass_evolution_derivative(
            self.prevIterationState.dt,
            self.prevIterationState.gamma,
            self.prevIterationState.prev_gamma,
        )

        change = [
            self.dt - self.prevIterationState.dt,
            self.gamma - self.prevIterationState.gamma,
            prev_gamma - self.prevIterationState.prev_gamma,
        ]
        return previousValue + self.math.dot(derivative, change)

    def base_position_evolution_derivative(self, dt, prev_vel_i, accel_i, prev_accel_i):
        return [
            prev_vel_i + (accel_i + prev_accel_i / 2) * 2 / 3 * dt,  # d(position)/d(dt)
            dt,  # d(position)/d(prev_vel_i)
            dt**2 / 3,  # d(position)/d(accel_i)
            dt**2 / 6,  # d(position)/d(prev_accel_i)
        ]

    def position_evolution_function(self, dt, prev_vel_i, accel_i, prev_accel_i, i):
        previousValue = self.base_position_evolution_function(
            dt=self.prevIterationState.dt,
            prev_vel_i=self.prevIterationState.prev_velocity[i],
            accel_i=self.prevIterationState.acceleration[i],
            prev_accel_i=self.prevIterationState.prev_acceleration[i],
        )

        derivative = self.base_position_evolution_derivative(
            self.prevIterationState.dt,
            self.prevIterationState.prev_velocity[i],
            self.prevIterationState.acceleration[i],
            self.prevIterationState.prev_acceleration[i],
        )

        change = [
            dt - self.prevIterationState.dt,
            prev_vel_i - self.prevIterationState.prev_velocity[i],
            accel_i - self.prevIterationState.acceleration[i],
            prev_accel_i - self.prevIterationState.prev_acceleration[i],
        ]
        return previousValue + self.math.dot(derivative, change)

    def base_velocity_evolution_derivative(self, dt, accel_i, prev_accel_i):
        return [
            (accel_i + prev_accel_i) / 2,  # d(velocity)/d(dt)
            dt / 2,  # d(velocity)/d(accel_i)
            dt / 2,  # d(velocity)/d(prev_accel_i)
        ]

    def velocity_evolution_function(self, dt, accel_i, prev_accel_i, i):
        previousValue = self.base_velocity_evolution_function(
            dt=self.prevIterationState.dt,
            accel_i=self.prevIterationState.acceleration[i],
            prev_accel_i=self.prevIterationState.prev_acceleration[i],
        )

        derivative = self.base_velocity_evolution_derivative(
            self.prevIterationState.dt,
            self.prevIterationState.acceleration[i],
            self.prevIterationState.prev_acceleration[i],
        )

        change = [
            dt - self.prevIterationState.dt,
            accel_i - self.prevIterationState.acceleration[i],
            prev_accel_i - self.prevIterationState.prev_acceleration[i],
        ]
        return previousValue + self.math.dot(derivative, change)

    def ComputeDragForce(self) -> Array3:
        prevItMag = np.linalg.norm(self.prevIterationState.velocity)
        return self.params.ComputeDragForce(self.velocity, prevItMag)

    def NewtonsSecondLaw(self, i):
        prevItMass = self.prevIterationState.mass
        return (
            prevItMass * (self.acceleration[i] + self.artificial_acceleration[i])
            == self.thrust[i] + self.drag_force[i] + self.params.g[i] * prevItMass
        )


class SequantialConvexification_Base_Model(pmo.block, ABC):
    def __init__(
        self,
        params: SystemParameters,
        nSteps: int,
        start: float,
        stop: Union[float, None],
    ):
        super().__init__()
        self.params = params
        self.start = start
        self.nSteps = nSteps

        if stop is None:
            self.variable_dt = True
            self.dt = pmo.variable(domain=pmo.NonNegativeReals, value=1.0)
            self.stop = self.start + self.nSteps * self.dt
        else:
            self.variable_dt = False
            self.dt = (stop - start) / nSteps
            self.stop = stop

        self.initState = State(
            mass=params.m0,
            position=params.x0,
            velocity=params.v0,
            acceleration=[0.0, 0.0, 0.0],
            thrust=params.T0,
            gamma=np.linalg.norm(params.T0),
        )

    def getIterationStates(self) -> Iterable[IterationState]:
        return [step.getIterationState() for step in self.steps]

    def Plot(self):
        # import matplotlib

        # matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        dt = pmo.value(self.dt)
        times = [self.start + i * dt for i in range(self.nSteps)]
        positions = np.array(
            [[pmo.value(step.position[i]) for step in self.steps] for i in range(3)]
        )
        velocities = np.array(
            [[pmo.value(step.velocity[i]) for step in self.steps] for i in range(3)]
        )
        accelerations = np.array(
            [[pmo.value(step.acceleration[i]) for step in self.steps] for i in range(3)]
        )
        masses = np.array([pmo.value(step.mass) for step in self.steps])

        fig = plt.figure(figsize=(12, 16))
        gs = fig.add_gridspec(3, 2)
        posAx = fig.add_subplot(gs[0, 0])
        velAx = fig.add_subplot(gs[0, 1])
        accAx = fig.add_subplot(gs[1, 0])
        massAx = fig.add_subplot(gs[1, 1])
        thrustAx = fig.add_subplot(gs[2, 0])
        from mpl_toolkits.mplot3d import Axes3D

        threeDimAx = fig.add_subplot(gs[2, 1], projection="3d")
        posAx.plot(times, positions[0, :], label="X")
        posAx.plot(times, positions[1, :], label="Y")
        posAx.plot(times, positions[2, :], label="Z")
        posAx.set_title("Position vs Time")
        posAx.set_xlabel("Time (s)")
        posAx.set_ylabel("Position (m)")
        posAx.legend()
        posAx.grid()

        velAx.plot(times, velocities[0, :], label="Vx")
        velAx.plot(times, velocities[1, :], label="Vy")
        velAx.plot(times, velocities[2, :], label="Vz")
        velAx.set_title("Velocity vs Time")
        velAx.set_xlabel("Time (s)")
        velAx.set_ylabel("Velocity (m/s)")
        velAx.legend()
        velAx.grid()

        accAx.plot(times, accelerations[0, :], label="Ax")
        accAx.plot(times, accelerations[1, :], label="Ay")
        accAx.plot(times, accelerations[2, :], label="Az")
        accAx.set_title("Acceleration vs Time")
        accAx.set_xlabel("Time (s)")
        accAx.set_ylabel("Acceleration (m/s²)")
        accAx.legend()
        accAx.grid()

        massAx.plot(times, masses, label="Mass", color="purple")
        massAx.axhline(self.params.m_dry, color="red", linestyle="--", label="Dry Mass")
        massAx.set_title("Mass vs Time")
        massAx.set_xlabel("Time (s)")
        massAx.set_ylabel("Mass (kg)")
        massAx.legend()
        massAx.grid()

        thrusts = np.array(
            [[pmo.value(step.thrust[i]) for step in self.steps] for i in range(3)]
        )
        thrustAx.plot(times, thrusts[0, :], label="Tx")
        thrustAx.plot(times, thrusts[1, :], label="Ty")
        thrustAx.plot(times, thrusts[2, :], label="Tz")
        thrustAx.set_title("Thrust vs Time")
        thrustAx.set_xlabel("Time (s)")
        thrustAx.set_ylabel("Thrust (N)")
        thrustAx.legend()
        thrustAx.grid()
        thrustAx.axhline(-self.params.T_max, color="red", linestyle="--")
        thrustAx.axhline(self.params.T_max, color="red", linestyle="--")

        threeDimAx.plot(
            positions[0, :], positions[1, :], positions[2, :], label="Trajectory"
        )
        nFigurines = 10
        figurineIndices = np.linspace(0, self.nSteps - 1, nFigurines).astype(int)
        for idx in figurineIndices:
            if idx == 0:
                kwargs = dict(label="Thrust Vector")
            else:
                kwargs = {}
            threeDimAx.quiver(
                positions[0, idx],
                positions[1, idx],
                positions[2, idx],
                thrusts[0, idx],
                thrusts[1, idx],
                thrusts[2, idx],
                length=np.linalg.norm(thrusts[:, idx]) / 500,
                normalize=True,
                color="red",
                pivot="middle",
                arrow_length_ratio=0.007,
                **kwargs,
            )

        xmin = min(min(positions[0, :]), self.params.xf[0])
        xmax = max(max(positions[0, :]), self.params.xf[0])
        ymin = min(min(positions[1, :]), self.params.xf[1])
        ymax = max(max(positions[1, :]), self.params.xf[1])
        zmin = min(min(positions[2, :]), self.params.xf[2])
        zmax = max(max(positions[2, :]), self.params.xf[2])
        xyMin = min(xmin, ymin)
        xyMax = max(xmax, ymax)
        x = np.linspace(xyMin, xyMax, 10)
        y = np.linspace(xyMin, xyMax, 10)
        X, Y = np.meshgrid(x, y)
        Z = (
            np.sin(self.params.max_glide_slope)
            * np.sqrt((X - self.params.xf[0]) ** 2 + (Y - self.params.xf[1]) ** 2)
            + self.params.xf[2]
        )
        threeDimAx.plot_surface(X, Y, Z, alpha=0.3, color="gray", label="Glide Cone")
        threeDimAx.set_title("3D Trajectory")
        threeDimAx.set_xlabel("X (m)")
        threeDimAx.set_ylabel("Y (m)")
        threeDimAx.set_zlabel("Z (m)")
        threeDimAx.legend()
        threeDimAx.grid()
        # threeDimAx.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))

        plt.tight_layout()
        plt.show()


class SequentialConvexification_Initial_Model(SequantialConvexification_Base_Model):
    def __init__(
        self,
        params: SequentialConvexification_Initial_Parameters,
        nSteps: int,
        start: float,
        stop: float,
        T_guess: Iterable[Array3] = None,
    ):
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

            step = SequentialConvexification_Initial_Step_Model(
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

        self.objective = pmo.objective(
            expr=(
                -params.w_m * self.steps[-1].mass
                + params.w_a * self.artificial_acceleration_norm
            ),
            sense=pmo.minimize,
        )


class SequentialConvexification_Iterate_Model(SequantialConvexification_Base_Model):
    def __init__(
        self,
        params: SequentialConvexification_Iterate_Parameters,
        nSteps: int,
        start: float,
    ):
        super().__init__(params, nSteps, start, stop=None)

        self.steps = pmo.block_list()
        for i in range(nSteps):
            isFinal = i == nSteps - 1
            prevState = self.initState if i == 0 else self.steps[i - 1].getState()

            step = SequentialConvexification_Iterate_Step_Model(
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

        self.objective = pmo.objective(
            expr=(
                -params.w_m * self.steps[-1].mass
                + params.w_eta_dt * self.eta_dt
                + params.w_eta_thrust * self.thrust_change_norm
                + params.w_a * self.artificial_acceleration_norm
            ),
            sense=pmo.minimize,
        )
