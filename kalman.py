"""Kalman filter example demo in Python.

A Python implementation of the example given in pages 11-15 of "An
Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
University of North Carolina at Chapel Hill, Department of Computer
Science, TR 95-041,
https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

by Andrew D. Straw

Re-adapted by Michael Otty
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    """Program starts here."""
    x, x_hat, z, error_estimate = kalman_filter()
    plot_everything(x, x_hat, z, error_estimate)


def plot_everything(x, x_hat, z, error_estimate):
    """Plot everything on a graph."""
    _, ax = plt.subplots(1, 1, sharex=True)

    ax.set_title("Estimate vs. iteration step")

    ax.plot(z, "k+", label="Noisy measurements", alpha=0.5)
    sigma = np.sqrt(error_estimate)
    two_sigma = 2 * sigma
    three_sigma = 3 * sigma

    ax.fill_between(
        range(0, len(x)),
        x_hat + three_sigma,
        x_hat - three_sigma,
        label=r"$3\sigma$",
    )
    ax.fill_between(
        range(0, len(x)),
        x_hat + two_sigma,
        x_hat - two_sigma,
        label=r"$2\sigma$",
    )
    ax.fill_between(
        range(0, len(x)),
        x_hat + sigma,
        x_hat - sigma,
        label=r"$1\sigma$",
    )
    ax.plot(x_hat, label="Estimate")
    ax.plot(x, "k-", label="Truth")

    ax.set_xlabel("Iteration")
    ax.legend()
    ax.grid()
    lim = 0.2
    ax.set_ylim(-lim, lim)
    ax.set_xlim(0, len(x))


def kalman_filter():
    """Simulate a Kalman filter with no process noise."""
    measurement_variance = 0.1

    # Initial parameters
    n_of_points = 500
    x = np.zeros(n_of_points)
    z = np.random.normal(0, np.sqrt(measurement_variance), size=n_of_points)
    z += x

    process_variance = 0

    x_estimate = np.zeros(n_of_points)
    error_estimate = np.zeros(n_of_points)

    # Initial guesses
    x_estimate[0] = 0
    error_estimate[0] = 1.0

    for k in range(1, n_of_points):
        model_variance = _calculate_model_variance(
            error_estimate[k - 1],
            process_variance,
        )

        kalman_gain = _calculate_kalman_gain(
            model_variance=model_variance,
            measurement_variance=measurement_variance,
        )

        x_estimate[k] = _calculate_x_estimate(
            last_x_estimate=x_estimate[k - 1],
            kalman_gain=kalman_gain,
            measurement=z[k],
        )
        error_estimate[k] = _calculate_error_estimate(
            kalman_gain=kalman_gain,
            model_variance=model_variance,
        )

    return x, x_estimate, z, error_estimate


def _calculate_model_variance(last_error_estimate, process_variance):
    return last_error_estimate + process_variance


def _calculate_kalman_gain(model_variance, measurement_variance):
    """Kalman gain of 1 is full trust of measurement, 0 if full trust of model."""
    return model_variance / (model_variance + measurement_variance)


def _calculate_x_estimate(last_x_estimate, kalman_gain, measurement):
    return last_x_estimate + kalman_gain * (measurement - last_x_estimate)


def _calculate_error_estimate(kalman_gain, model_variance):
    return (1 - kalman_gain) * model_variance


if __name__ == "__main__":
    main()
    plt.show()
