"""
WindGenerator
=============

This module defines the `Wind_TimeSeries` and `Wind_Function` classes, which model general time series for wind dynamics, including magnitudes and angles.
"""

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from scipy.interpolate import CubicSpline as interp1d


from random import Random

wind_rng = Random()


class Wind_TimeSeries:
    """
    Wind_TimeSeries
    ----------------

    Models a time series of wind magnitudes or angles using Gaussian processes.

    Attributes:
        correlation_coef (float): Coefficient for magnitude-to-variance conversion.
        variance (float): Variance of the wind magnitude or angle.
        magnitude (float): Magnitude of the wind or angle.
        bias (float): Bias added to the wind magnitude or angle.
        length_scale (float): Length scale for the Gaussian process kernel.
        start_time (float): Start time of the time series.
        end_time (float): End time of the time series.
        ts (np.ndarray): Time points for the time series.
        values (np.ndarray): Wind magnitudes or angles at the time points.
        func (callable): Interpolated function for the time series.
    """

    correlation_coef = 2.45

    def __init__(
        self,
        end_time,
        start_time=0.0,
        variance=None,
        magnitude=None,
        bias=0.0,
        random_state=None,
        length_scale=15.0,
        resolution=100,
        training_points=None,
    ):
        """
        Initializes the `Wind_TimeSeries` with the given parameters.

        Args:
            end_time (float): End time of the time series.
            start_time (float, optional): Start time of the time series. Defaults to 0.0.
            variance (float, optional): Variance of the wind magnitude. Defaults to None.
            magnitude (float, optional): Magnitude of the wind. Defaults to None.
            bias (float, optional): Bias added to the wind magnitude. Defaults to 0.0.
            random_state (int, optional): Random state for reproducibility. Defaults to None.
            length_scale (float, optional): Length scale for the Gaussian process kernel. Defaults to 15.0.
            resolution (int, optional): Resolution of the time series. Defaults to 100.
            training_points (np.ndarray, optional): Training points for the Gaussian process. Defaults to None.

        Raises:
            ValueError: If both `variance` and `magnitude` are specified or neither is specified.
        """

        if (variance is None and magnitude is None) or (
            variance is not None and magnitude is not None
        ):
            raise ValueError(
                "Either variance or magnitude must be specified, but not both."
            )

        if magnitude is not None:
            self.variance = (magnitude / self.correlation_coef) ** 2
            self.magnitude = magnitude
        else:
            self.variance = variance
            self.magnitude = self.correlation_coef * np.sqrt(variance)

        self.length_scale = length_scale
        self.bias = bias

        if random_state is None:
            self.random_state = self.get_random_state()
        else:
            self.random_state = random_state

        kernel = self.variance * Matern(length_scale=length_scale, nu=1.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, random_state=self.random_state
        )

        if training_points is not None:
            t_train = training_points[:, 0].reshape(-1, 1)
            mag_train = training_points[:, 1].reshape(-1, 1)
            self.gp.fit(t_train, mag_train)

        self.start_time = start_time
        self.end_time = end_time

        self.ts = np.linspace(self.start_time, self.end_time, resolution).reshape(-1, 1)
        self.values = self.gp.sample_y(self.ts, random_state=self.random_state)

        self.func = interp1d(self.ts.flatten(), self.values.flatten())

    def get_random_state(self):
        return wind_rng.randint(0, 4294967295)

    def __call__(self, t, include_bias=True):
        if include_bias:
            return self.func(t) + self.bias
        else:
            return self.func(t)

    def spawn(
        self,
        copy_end,
        copy_start=None,
        new_start_time=None,
        new_end_time=None,
        resolution=100,
        random_state=None,
    ):
        """
        Creates a new `Wind_TimeSeries` instance with variations for stochastic modeling.

        Args:
            copy_end (float): End time for the copied time series.
            copy_start (float, optional): Start time for the copied time series. Defaults to None.
            new_start_time (float, optional): Start time for the new time series. Defaults to None.
            new_end_time (float, optional): End time for the new time series. Defaults to None.
            resolution (int, optional): Resolution of the new time series. Defaults to 100.
            random_state (int, optional): Random state for reproducibility. Defaults to None.

        Returns:
            Wind_TimeSeries: A new instance with variations applied.
        """

        if copy_start is None:
            copy_start = self.start_time
        if new_start_time is None:
            new_start_time = self.start_time
        if new_end_time is None:
            new_end_time = self.end_time

        assert copy_start < copy_end, "copy_start must be less than copy_end"
        assert (
            copy_start >= self.start_time
        ), "copy_start must be >= original start_time"
        assert copy_end <= self.end_time, "copy_end must be <= original end_time"

        t_train = np.linspace(copy_start, copy_end, resolution)
        mag_train = self(t_train, include_bias=False)
        training_points = np.vstack((t_train, mag_train)).T

        if random_state is None:
            random_state = self.get_random_state()

        return Wind_TimeSeries(
            start_time=new_start_time,
            end_time=new_end_time,
            variance=self.variance,
            random_state=random_state,
            length_scale=self.length_scale,
            resolution=resolution,
            training_points=training_points,
        )


class Wind_Function:
    """
    Wind_Function
    ------------

    Models wind dynamics in spherical coordinates using `Wind_TimeSeries` for magnitudes and angles.

    Attributes:
        magnitude (float): Magnitude of the wind.
        start_time (float): Start time of the wind function.
        end_time (float): End time of the wind function.
        magnitude_series (Wind_TimeSeries): Time series for wind magnitude.
        theta_series (Wind_TimeSeries): Time series for azimuth angle.
        phi_series (Wind_TimeSeries): Time series for elevation angle.
    """

    def __init__(
        self,
        magnitude,
        bias,
        end_time,
        start_time=0.0,
        random_state=None,
        resolution=100,
        training_points=None,
    ):
        """
        Initializes the `Wind_Function` with the given parameters.

        Args:
            magnitude (float): Magnitude of the wind.
            bias (float): Bias added to the wind magnitude.
            end_time (float): End time of the wind function.
            start_time (float, optional): Start time of the wind function. Defaults to 0.0.
            random_state (int, optional): Random state for reproducibility. Defaults to None.
            resolution (int, optional): Resolution of the wind function. Defaults to 100.
            training_points (np.ndarray, optional): Training points for the wind function. Defaults to None.
        """

        self.magnitude = magnitude
        self.start_time = start_time
        self.end_time = end_time

        if random_state is None:
            self.random_state = self.get_random_state()
        else:
            self.random_state = random_state

        if training_points is not None:
            training_mag = training_points[:, [0, 1]]
            training_theta = training_points[:, [0, 2]]
            training_phi = training_points[:, [0, 3]]
        else:
            training_mag = None
            training_theta = None
            training_phi = None

        self.magnitude_series = Wind_TimeSeries(
            magnitude=magnitude,
            bias=bias,
            start_time=start_time,
            end_time=end_time,
            random_state=self.random_state,
            length_scale=15.0,
            resolution=resolution,
            training_points=training_mag,
        )
        self.random_state += 1

        self.theta_series = Wind_TimeSeries(
            magnitude=np.pi,
            start_time=start_time,
            end_time=end_time,
            random_state=self.random_state,
            length_scale=60.0,
            resolution=resolution,
            training_points=training_theta,
        )
        self.random_state += 1

        self.phi_series = Wind_TimeSeries(
            magnitude=np.pi
            / 6,  # Wind blowing directly up or down is unlikely, so we limit the vertical component to be smaller
            bias=np.pi / 2,  # Center around pi/2 (horizontal wind)
            start_time=start_time,
            end_time=end_time,
            random_state=self.random_state,
            length_scale=60.0,
            resolution=resolution,
            training_points=training_phi,
        )
        self.random_state += 1

    def spawn(
        self,
        copy_end,
        copy_start=None,
        new_start_time=None,
        new_end_time=None,
        resolution=100,
        random_state=None,
    ):
        """
        Creates a new `Wind_Function` instance with variations for stochastic modeling.

        Args:
            copy_end (float): End time for the copied wind function.
            copy_start (float, optional): Start time for the copied wind function. Defaults to None.
            new_start_time (float, optional): Start time for the new wind function. Defaults to None.
            new_end_time (float, optional): End time for the new wind function. Defaults to None.
            resolution (int, optional): Resolution of the new wind function. Defaults to 100.
            random_state (int, optional): Random state for reproducibility. Defaults to None.

        Returns:
            Wind_Function: A new instance with variations applied.
        """

        if copy_start is None:
            copy_start = self.start_time
        if new_start_time is None:
            new_start_time = self.start_time
        if new_end_time is None:
            new_end_time = self.end_time

        assert copy_start < copy_end, "copy_start must be less than copy_end"
        assert (
            copy_start >= self.start_time
        ), "copy_start must be >= original start_time"
        assert copy_end <= self.end_time, "copy_end must be <= original end_time"

        if random_state is None:
            random_state = self.get_random_state()

        time_series = np.linspace(copy_start, copy_end, resolution)
        mag_train = self.magnitude_series(time_series, include_bias=False)
        theta_train = self.theta_series(time_series, include_bias=False)
        phi_train = self.phi_series(time_series, include_bias=False)
        training_points = np.vstack((time_series, mag_train, theta_train, phi_train)).T

        return Wind_Function(
            magnitude=self.magnitude,
            bias=self.magnitude_series.bias,
            start_time=new_start_time,
            end_time=new_end_time,
            random_state=random_state,
            resolution=resolution,
            training_points=training_points,
        )

    def evaluate_spherical(self, t, include_bias=True):
        """
        Evaluates the wind function in spherical coordinates.

        Args:
            t (float): Time at which to evaluate the wind function.
            include_bias (bool, optional): Whether to include the bias in the evaluation. Defaults to True.

        Returns:
            tuple: Magnitude, azimuth, and elevation of the wind at time `t`.
        """

        mag = self.magnitude_series(t, include_bias=include_bias)
        theta = self.theta_series(t, include_bias=include_bias)
        phi = self.phi_series(t, include_bias=include_bias)
        return mag, theta, phi

    def __call__(self, t, include_bias=True):
        """
        Evaluates the wind function in Cartesian coordinates.

        Args:
            t (float): Time at which to evaluate the wind function.
            include_bias (bool, optional): Whether to include the bias in the evaluation. Defaults to True.

        Returns:
            np.ndarray: Wind vector in Cartesian coordinates at time `t`.
        """

        mag, theta, phi = self.evaluate_spherical(t, include_bias=include_bias)

        x = mag * np.sin(theta) * np.cos(phi)
        y = mag * np.sin(theta) * np.sin(phi)
        z = mag * np.cos(phi)

        return np.vstack((x, y, z)).T

    def get_random_state(self):
        return wind_rng.randint(0, 4294967295)

    def Plot(self, axes=None):
        import matplotlib.pyplot as plt

        if axes is None:
            fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        xAx, yAx, zAx = axes
        times = np.linspace(self.start_time, self.end_time, 100)
        ws = self(times)
        xAx.plot(times, ws[:, 0], color="green")
        yAx.plot(times, ws[:, 1], color="orange")
        zAx.plot(times, ws[:, 2], color="blue")


if __name__ == "__main__":
    # EXAMPLE USAGE
    import matplotlib.pyplot as plt

    rand = None
    wind = Wind_Function(
        magnitude=3, bias=5, start_time=0, end_time=200, random_state=rand
    )

    ts = np.linspace(0, 200, 100)
    ws = wind(ts)

    wind2 = wind.spawn(copy_start=50, copy_end=150, new_start_time=0, new_end_time=200)
    ws2 = wind2(ts)

    # 2-D Plots:
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    ax[0].plot(ts, ws[:, 0])
    ax[1].plot(ts, ws[:, 1])
    ax[2].plot(ts, ws[:, 2])

    ax[0].plot(ts, ws2[:, 0], linestyle="--")
    ax[1].plot(ts, ws2[:, 1], linestyle="--")
    ax[2].plot(ts, ws2[:, 2], linestyle="--")

    ax[0].set_ylabel("Wind X (km/s)")
    ax[1].set_ylabel("Wind Y (km/s)")
    ax[2].set_ylabel("Wind Z (km/s)")
    ax[2].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    # 3-D Animation:
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.set_xlim([np.min(ws[:, 0]), np.max(ws[:, 0])])
    # ax.set_ylim([np.min(ws[:, 1]), np.max(ws[:, 1])])
    # ax.set_zlim([np.min(ws[:, 2]), np.max(ws[:, 2])])
    # ax.set_xlabel("X (km/s)")
    # ax.set_ylabel("Y (km/s)")
    # ax.set_zlabel("Z (km/s)")
    # ax.set_title("Wind Vector Animation (Magnitude = 10 km/s)")

    # quiver = None  # Initialize quiver as a global variable

    # # Initial vector
    # quiver = ax.quiver(
    #     0, 0, 0, ws[0, 0], ws[0, 1], ws[0, 2], color="r", length=1, normalize=True
    # )

    # def animate(i):
    #     global quiver  # Access the global quiver object
    #     if quiver is not None:
    #         quiver.remove()  # Remove the previous quiver
    #     quiver = ax.quiver(
    #         0, 0, 0, ws[i, 0], ws[i, 1], ws[i, 2], color="r", length=1, normalize=True
    #     )
    #     return (quiver,)

    # ani = animation.FuncAnimation(fig, animate, frames=len(ts), interval=50, blit=False)
    # plt.show()
