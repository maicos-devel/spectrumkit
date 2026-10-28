#!/usr/bin/env python
r""".. _howto-debye-relaxation:

Debye relaxation of an Ornstein-Uhlenbeck dipole
------------------------------------------------

The test suite checks the spectrum routes against a dipole whose susceptibility is known
exactly. This example walks through that test, ``TestDebyeRelaxation`` in
``tests/tests.py``, and shows what it compares.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from scipy.signal import lfilter

from spectrumkit import calculate_spectrum_from_current, calculate_spectrum_from_dipole

BLUE, ORANGE, INK, SHADE = "#2a78d6", "#eb6834", "#0b0b0b", "#e1e0d9"

# sphinx_gallery_thumbnail_number = 2

# %%
# The model
# ~~~~~~~~~
#
# Each Cartesian component of the dipole :math:`\boldsymbol{M}` follows the Langevin
# equation
#
# .. math::
#
#     \dot M_\alpha(t) = -\frac{M_\alpha(t)}{\tau}
#     + \sqrt{\frac{2 \sigma^2}{\tau}} \, \xi_\alpha(t),
#     \qquad \langle \xi_\alpha(t) \xi_\beta(t') \rangle
#     = \delta_{\alpha \beta} \, \delta(t - t').
#
# The equation is linear and the noise is uncorrelated with the past, so multiplying by
# :math:`M_\alpha(0)` and averaging leaves :math:`\dot C = -C / \tau` for the
# autocorrelation:
#
# .. math::
#
#     C(t) = \langle \boldsymbol{M}(0) \cdot \boldsymbol{M}(t) \rangle
#     = 3 \sigma^2 e^{-|t| / \tau}.
#
# The classical fluctuation-dissipation theorem turns this into the Debye
# susceptibility of a system with volume :math:`V` at temperature :math:`T`,
#
# .. math::
#
#     \chi(\omega) = \frac{\chi_0}{1 - i \omega \tau}, \qquad
#     \chi''(\omega) = \chi_0 \, \frac{\omega \tau}{1 + \omega^2 \tau^2}, \qquad
#     \chi_0 = \frac{\langle \boldsymbol{M}^2 \rangle}{3 V \varepsilon_0 k_B T},
#
# with the loss peak at :math:`\omega \tau = 1`.
#
# Generating the dipole
# ~~~~~~~~~~~~~~~~~~~~~
#
# Integrating the Langevin equation over one time step :math:`\Delta t` gives the
# recursion
#
# .. math::
#
#     M_{n+1} = a M_n + \sigma \sqrt{1 - a^2} \, \eta_n, \qquad
#     a = e^{-\Delta t / \tau},
#
# with standard normal :math:`\eta_n`. It is exact for any time step, and drawing the
# first frame from the stationary distribution avoids an equilibration phase. The test
# uses 2 ns of a dipole with a relaxation time of 1 ps, sampled every 10 fs. (The test
# suite generates it with the helper ``ornstein_uhlenbeck`` in
# ``tests/data/synthetic.py``.)

dt = 0.01  # ps
tau = 1.0  # ps
sigma = 1.0  # e·Å
volume = 1000.0  # Å^3
temperature = 300.0  # K
n_frames = 200_000

rng = np.random.default_rng(0)
a = np.exp(-dt / tau)
noise = rng.normal(0, sigma * np.sqrt(1 - a**2), (n_frames, 3))
noise[0] = rng.normal(0, sigma, 3)
dipole = lfilter([1], [1, -a], noise, axis=0)

# %%
# The trajectory looks like noise, but its autocorrelation decays as the model predicts.

lags = np.arange(0, int(4 * tau / dt), 5)
acf = np.array(
    [np.mean(np.sum(dipole[: n_frames - k] * dipole[k:], axis=1)) for k in lags]
)

time = dt * np.arange(n_frames)
shown = time < 20

fig, (ax_traj, ax_acf) = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
ax_traj.plot(time[shown], dipole[shown, 0], color=BLUE, lw=1)
ax_traj.set(xlabel="$t$ / ps", ylabel=r"$M_x$ / e$\cdot$Å", title="Trajectory")

ax_acf.plot(lags * dt / tau, acf / acf[0], "o", color=BLUE, ms=4, label="trajectory")
ax_acf.plot(
    lags * dt / tau, np.exp(-lags * dt / tau), color=INK, label=r"$e^{-t/\tau}$"
)
ax_acf.set(xlabel=r"$t / \tau$", ylabel="$C(t) / C(0)$", title="Autocorrelation")
ax_acf.legend(frameon=False)
plt.show()

# %%
# Spectra from the dipole and from the current
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :func:`~spectrumkit.dielectricspectrum.calculate_spectrum_from_dipole` works on the
# dipole, :func:`~spectrumkit.dielectricspectrum.calculate_spectrum_from_current` on the
# current :math:`\boldsymbol{J} = \dot{\boldsymbol{M}}`, here a finite difference. Both
# average the periodogram over 100 segments of 20 ps.

dipole_route = calculate_spectrum_from_dipole(dipole, dt, volume, temperature, segs=100)
current = np.diff(dipole, axis=0) / dt
current_route = calculate_spectrum_from_current(
    current, dt, volume, temperature, segs=100
)

# %%
# The reference is the Debye susceptibility. :math:`\chi_0` uses the variance of the
# generated dipole, so the comparison checks the shape and normalization of the
# spectrum, not the sampling error of :math:`\langle \boldsymbol{M}^2 \rangle`.

pref = scipy.constants.e**2 * scipy.constants.angstrom**2
pref /= 3 * volume * scipy.constants.angstrom**3
pref /= scipy.constants.k * temperature * scipy.constants.epsilon_0
chi_0 = pref * np.sum(np.var(dipole, axis=0))


def debye(nu):
    """Debye susceptibility at the frequency ``nu`` in THz."""
    return chi_0 / (1 - 2j * np.pi * nu * tau)


# %%
# The test compares the imaginary part in the band :math:`0.5 < \omega \tau < 2` around
# the loss peak (shaded) and passes when the mean ratio to the Debye curve is within
# 5 % of one (dotted lines).

band = np.array([0.5, 2]) / (2 * np.pi * tau)
nu = np.logspace(-2, np.log10(50), 300)

fig, (ax, ax_ratio) = plt.subplots(
    2, 1, sharex=True, figsize=(7, 5.5), height_ratios=(2, 1), layout="constrained"
)
ax.loglog(nu, debye(nu).imag, color=INK, ls="--", label="Debye")
for res, label, color in [
    (dipole_route, "from dipole", BLUE),
    (current_route, "from current", ORANGE),
]:
    nu_res, chi_res = res["nu_binned"][1:], res["susc_binned"].imag[1:]
    ax.loglog(nu_res, chi_res, color=color, lw=1.5, label=label)
    ax_ratio.semilogx(nu_res, chi_res / debye(nu_res).imag, color=color, lw=1.5)

for axis in (ax, ax_ratio):
    axis.axvspan(*band, color=SHADE, zorder=0)
ax_ratio.axhline(1, color=INK, ls="--", lw=1)
for limit in (0.95, 1.05):
    ax_ratio.axhline(limit, color=INK, ls=":", lw=1)

ax.set(ylabel=r"$\chi''$", title="Dielectric loss")
ax.legend(frameon=False)
ax_ratio.set(xlabel=r"$\nu$ / THz", ylabel=r"$\chi'' / \chi''_\mathrm{Debye}$")
ax_ratio.set_ylim(0.6, 1.4)
plt.show()

# %%
# Inside the band, the mean ratios are

for name, res in [("dipole", dipole_route), ("current", current_route)]:
    in_band = (res["nu"] > band[0]) & (res["nu"] < band[1])
    ratio = np.mean(res["susc"].imag[in_band] / debye(res["nu"][in_band]).imag)
    print(f"from {name}: {ratio:.3f}")

# %%
# Outside the band, the curves deviate in ways the test leaves out:
#
# - Above about 10 THz the dipole route rises. This is aliasing, not an error of the
#   route: sampled every 10 fs, the series has a higher spectrum near the Nyquist
#   frequency of 50 THz than the continuous process.
# - Between 1 and 10 THz the dipole route lies about 5 % high, and the current route
#   overshoots at its lowest frequencies. Both are spectral leakage: the periodogram
#   treats each 20 ps segment as periodic, and the jump between its ends adds a
#   spurious contribution.
#
# The test runs the same check for
# :func:`~spectrumkit.dielectricspectrum.WienerKhinchin` with
# ``spectrum_type="dipole"`` and ``"flux"``. It covers only the imaginary part; the real
# part comes from a numerical Kramers-Kronig transform and is not tested here.
