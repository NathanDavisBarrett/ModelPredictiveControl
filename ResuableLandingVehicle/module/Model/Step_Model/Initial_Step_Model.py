from .Base_Step_Model import Base_Step_Model
from ...System import SystemParameters
from ...Util.Math import Array3
from ..State import State, InitializationState


class Initial_Step_Model(Base_Step_Model):
    def __init__(
        self,
        t_est: float,  # Estimated time (used for determining wind speed)
        params: SystemParameters,
        dt: float,  # Float must be fixed for the initial guess
        reference_mass: float = None,
        reference_speed: Array3 = None,
        prevState: State = None,
        isFinal: bool = False,
        initializationState: InitializationState = None,
    ):

        super().__init__(
            t_est=t_est,
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
        v_eff = self.math.vector_add(self.velocity, self.wind_velocity)
        return self.params.ComputeDragForce(v_eff, v_mag=self.reference_speed)

    def NewtonsSecondLaw(self, i):
        # Mg * (km/s^2 + km/s^2) = kN + kN + (km/s^2 * Mg)
        # Mg * (km/s^2) = kN + (km/s^2 * Mg)
        # (kN / (kg * km/s^2)) * (Mg) * (km/s^2) = kN + (kN / (kg * km/s^2)) * (Mg) * (km/s^2)
        # kN * (Mg/kg) = kN + kN * (Mg/kg)
        # 1000 kN = kN + 1000 kN
        return (
            self.reference_mass
            * 1000
            * (self.acceleration[i] + self.artificial_acceleration[i])
            == self.thrust[i]
            + self.drag_force[i]
            + self.params.g[i]
            * self.reference_mass
            * 1000  # TODO: Replace 2nd mass term with variable mass
        )
