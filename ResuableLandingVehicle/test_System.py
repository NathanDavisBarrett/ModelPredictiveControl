from System import System, SystemParameters, NumpyMath

import numpy as np
import warnings


def test_system_step():
    params = SystemParameters()
    params.math = NumpyMath()
    system = System(params)

    T = params.T0 * 1.1
    dt = 10

    system.step(T, dt)


def perform_simulation(
    alpha, beta, gamma, m0, numStep=30, dt=0.5, use_linearized_model: bool = False
):
    """
    Performs a simulation of the system dynamics with a thrust profile defined by

        T(t) = Z_unit * (alpha * t^2 + beta * t + gamma)

    where Z_unit is the upward-facing unit vector.

    Args:
        alpha: Quadratic coefficient of the thrust profile (N/s^2)
        beta: Linear coefficient of the thrust profile (N/s)
        gamma: Constant coefficient of the thrust profile (N)
        numStep: Number of simulation steps
        dt: Time step duration (s)

    Returns:
        xs: Array of positions at each time step (numStep x 3)
        vs: Array of velocities at each time step (numStep x 3)
        ms: Array of masses at each time step (numStep)
    """
    params = SystemParameters()
    params.m0 = m0
    params.T0 = np.array([0, 0, gamma])
    params.math = NumpyMath()
    if use_linearized_model:
        start = 0
        stop = numStep * dt
        initialSpeed = params.math.norm(params.v0)
        finalSpeed = params.math.norm(params.vf)
        dsdt = (finalSpeed - initialSpeed) / (stop - start)
        dmdt = (params.m_dry - params.m0) / (stop - start)

        system = System(params, reference_mass=params.m0, reference_speed=initialSpeed)
    else:
        system = System(params)

    xs = np.empty((numStep, 3))
    xs[0, :] = system.x

    vs = np.empty((numStep, 3))
    vs[0, :] = system.v

    ms = np.empty(numStep)
    ms[0] = system.m

    Ts = np.empty((numStep, 3))
    Ts[0, :] = params.T0

    z_unit = np.array([0, 0, 1])
    for i in range(1, numStep):
        t = i * dt
        T = z_unit * (alpha * t**2 + beta * t + gamma)
        Ts[i, :] = T

        if use_linearized_model:
            system.reference_mass = params.m0 + dmdt * t
            system.reference_speed = initialSpeed + dsdt * t

        system.step(T, dt)
        xs[i, :] = system.x
        vs[i, :] = system.v
        ms[i] = system.m

    return xs, vs, ms, Ts


def solve_coefs(coefs, use_linearized_model: bool = False):
    m0, beta, gamma = coefs
    alpha = 0
    # print(f"Testing alpha={alpha}, beta={beta}, gamma={gamma}")
    xs, vs, ms, _ = perform_simulation(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        m0=m0,
        numStep=120,
        dt=0.25,
        use_linearized_model=use_linearized_model,
    )
    final_pos = xs[-1, 2]
    final_vel = vs[-1, 2]
    final_mass = ms[-1]

    result = final_pos, final_vel, final_mass - SystemParameters().m_dry * 1.1
    # print(f"Final pos: {final_pos}, vel: {final_vel}, mass: {final_mass}")
    # print("-----")
    return result


def test_graphic():
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(321, projection="3d")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 2])
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")

    numStep = 120
    dt = 0.25

    alpha = 0
    beta = 2  # kN/s
    gamma = 1  # kN
    m0 = 15  # Mg

    from scipy.optimize import fsolve

    use_linearized_model = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m0, beta, gamma = fsolve(
            solve_coefs, [m0, beta, gamma], args=(use_linearized_model,)
        )

    xs, vs, ms, Ts = perform_simulation(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        m0=m0,
        numStep=numStep,
        dt=dt,
        use_linearized_model=use_linearized_model,
    )
    print(Ts.shape)
    np.save("landing_thrust.npy", Ts)
    print("m0 = ", m0, "Mg")

    ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], "b-")

    ax2 = fig.add_subplot(322)
    ts = np.arange(numStep) * dt
    ax2.plot(ts, xs[:, 2], "b-")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Altitude (km)")

    ax3 = fig.add_subplot(323)
    ax3.plot(ts, vs[:, 2], "b-")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Vertical Speed (km/s)")

    ax4 = fig.add_subplot(324)
    Ts = alpha * ts**2 + beta * ts + gamma
    ax4.plot(ts, Ts, "b-")
    ax4.axhline(SystemParameters().T_min, color="r", linestyle="--")
    ax4.axhline(SystemParameters().T_max, color="r", linestyle="--")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Thrust Magnitude (kN)")

    ax5 = fig.add_subplot(325)
    ax5.plot(ts, ms, "b-")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Mass (Mg)")
    ax5.axhline(SystemParameters().m_dry, color="r", linestyle="--")

    plt.show()


if __name__ == "__main__":
    test_graphic()
