#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic time series with analytically known spectra."""

import numpy as np
from scipy.signal import lfilter


def ornstein_uhlenbeck(
    n_frames: int,
    n_dim: int = 3,
    dt: float = 1.0,
    tau: float = 1.0,
    sigma: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    r"""Ornstein-Uhlenbeck process, e.g. a dipole with Debye relaxation.

    Each of the ``n_dim`` independent components has zero mean, variance
    ``sigma**2`` and an exponential autocorrelation with correlation time ``tau``.
    Its spectra are known in closed form (see Notes), which makes it a reference for
    testing spectrum estimators. The process is sampled exactly as an AR(1) recursion
    and starts from the stationary distribution.

    Parameters
    ----------
    n_frames : int
        Number of time steps.
    n_dim : int, default=3
        Number of independent components.
    dt : float, default=1.0
        Time step.
    tau : float, default=1.0
        Correlation time, in the same unit as ``dt``.
    sigma : float, default=1.0
        Standard deviation of each component.
    seed : int, default=0
        Seed of the random number generator.

    Returns
    -------
    numpy.ndarray
        Time series with shape (n_frames, n_dim).

    Notes
    -----
    Each component solves the Langevin equation

    .. math::

        \dot x(t) = -\frac{x(t)}{\tau} + \sqrt{\frac{2 \sigma^2}{\tau}} \, \xi(t),
        \qquad \langle \xi(t) \xi(t') \rangle = \delta(t - t').

    The equation is linear and the noise is uncorrelated with the past, so multiplying
    by :math:`x(0)` and averaging leaves :math:`\dot C = -C / \tau` for the
    autocorrelation, i.e. :math:`C(t) = \sigma^2 e^{-|t| / \tau}`. Its Fourier
    transform is a Lorentzian, and the fluctuation-dissipation theorem turns it into
    the Debye susceptibility :math:`\chi_0 / (1 - i \omega \tau)`.

    """
    rng = np.random.default_rng(seed)
    a = np.exp(-dt / tau)
    noise = rng.normal(0, sigma * np.sqrt(1 - a**2), (n_frames, n_dim))
    noise[0] = rng.normal(0, sigma, n_dim)  # start from the stationary distribution
    return lfilter([1], [1, -a], noise, axis=0)
