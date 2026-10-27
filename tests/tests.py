# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DielectricSpectrum class."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
import scipy.constants
from numpy.testing import assert_allclose

from spectrumkit import (
    DielectricSpectrum,
    WienerKhinchin,
    calculate_spectrum_from_current,
    calculate_spectrum_from_dipole,
)

sys.path.append(str(Path(__file__).parents[1]))
from data import WATER_TPR_NPT, WATER_TRR_NPT
from data.synthetic import ornstein_uhlenbeck


class TestDielectricSpectrum:
    """Tests for the DielectricSpectrum class."""

    @pytest.fixture
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        return u.atoms

    def test_output_name(self, ag, monkeypatch, tmp_path):
        """Test output name."""
        monkeypatch.chdir(tmp_path)

        ds = DielectricSpectrum(ag)
        ds.run()
        ds.save()
        with Path("susc.dat").open():
            pass
        with Path("P_tseries.npy").open():
            pass
        with Path("tseries.npy").open():
            pass
        with Path("V.txt").open():
            pass

    def test_output_name_prefix(self, ag, monkeypatch, tmp_path):
        """Test output name with custom prefix."""
        monkeypatch.chdir(tmp_path)

        ds = DielectricSpectrum(ag, output_prefix="foo")
        ds.run()
        ds.save()
        with Path("foo_susc.dat").open():
            pass
        with Path("foo_P_tseries.npy").open():
            pass
        with Path("foo_tseries.npy").open():
            pass
        with Path("foo_V.txt").open():
            pass

    def test_output_name_binned(self, ag, monkeypatch, tmp_path):
        """Test output name of binned data."""
        """
        The parameters are not meant to be sensible,
        but just to force the binned output.
        """
        monkeypatch.chdir(tmp_path)

        ds = DielectricSpectrum(ag, bins=5, binafter=0, segs=5)
        ds.run()
        ds.save()
        with Path("susc.dat").open():
            pass
        with Path("susc_binned.dat").open():
            pass
        with Path("P_tseries.npy").open():
            pass
        with Path("tseries.npy").open():
            pass
        with Path("V.txt").open():
            pass

    def test_output(self, ag, monkeypatch, tmp_path):
        """Test output values by comparing with magic numbers."""
        monkeypatch.chdir(tmp_path)

        ds = DielectricSpectrum(ag)
        ds.run()

        V = 1559814.4
        nu = [0.0, 0.5, 1.0]
        susc = [1.35 + 0.0j, 2.15 + 2.830074j, 1.35 + 2.8j]
        dsusc = [0.25 + 0.0j, 0.35 + 0.3829775j, 0.25 + 0.45j]

        assert_allclose(ds.V, V, rtol=1e-1)
        assert_allclose(ds.results.nu, nu, rtol=1)
        assert_allclose(ds.results.susc, susc, rtol=1e-1)
        assert_allclose(ds.results.dsusc, dsusc, rtol=1e-1)


class TestCalculateSpectrumFromDipole:
    """Tests for the standalone calculate_spectrum_from_dipole function."""

    def test_basic_calculation(self):
        """Test basic spectrum calculation from synthetic dipole data."""
        # Create synthetic dipole data
        n_frames = 100
        dipole_moment = np.random.randn(n_frames, 3) * 10
        dt = 2.0  # ps
        volume = 1000.0  # Ų
        temperature = 300.0  # K

        results = calculate_spectrum_from_dipole(
            dipole_moment=dipole_moment,
            dt=dt,
            volume=volume,
            temperature=temperature,
            segs=5,
            nobin=True,
        )

        # Check that results contain expected keys
        assert "nu" in results
        assert "susc" in results
        assert "dsusc" in results
        assert "t" in results

        # Check shapes and types
        assert isinstance(results["nu"], np.ndarray)
        assert isinstance(results["susc"], np.ndarray)
        assert isinstance(results["dsusc"], np.ndarray)
        assert results["susc"].dtype == complex
        assert results["dsusc"].dtype == complex

    def test_binning_without_unbinned_points(self):
        """Test logarithmic binning when no low-frequency points stay unbinned."""
        rng = np.random.default_rng(0)
        results = calculate_spectrum_from_dipole(
            dipole_moment=rng.standard_normal((1000, 3)),
            dt=1.0,
            volume=1000.0,
            temperature=300.0,
            segs=1,
            bins=5,
            binafter=0,
        )

        # the zero frequency stays on its own, the rest is binned logarithmically
        assert results["nu_binned"][0] == results["nu"][0]
        assert 1 < len(results["nu_binned"]) <= 6

    def test_consistency_with_dielectric_spectrum(self, monkeypatch, tmp_path):
        """Test that standalone function gives same results as class method."""
        monkeypatch.chdir(tmp_path)

        # Load test data
        u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        ag = u.atoms

        # Run DielectricSpectrum
        ds = DielectricSpectrum(ag, segs=1, nobin=True)
        ds.run()

        # Use standalone function with extracted dipole data
        results = calculate_spectrum_from_dipole(
            dipole_moment=ds.results.P,
            dt=ds.dt,
            volume=ds.results.V,
            temperature=ds.temperature,
            segs=1,
            nobin=True,
        )

        # Results should be very similar
        assert_allclose(ds.results.nu, results["nu"], rtol=1e-10)
        assert_allclose(ds.results.susc, results["susc"], rtol=1e-10)
        assert_allclose(ds.results.dsusc, results["dsusc"], rtol=1e-10)


class TestDebyeRelaxation:
    """Compare the spectrum routes to an Ornstein-Uhlenbeck dipole.

    A dipole with exponential autocorrelation <P(0)P(t)> = <P²> exp(-t/τ) has the
    Debye susceptibility χ(ω) = χ_0 / (1 - iωτ) with χ_0 = <P²> / (3 V k_B T ε_0).
    """

    dt = 0.01  # ps
    tau = 1.0  # ps
    volume = 1000.0  # Å^3
    temperature = 300.0  # K

    @pytest.fixture(scope="class")
    def dipole(self):
        """Ornstein-Uhlenbeck dipole time series (n_frames, 3) in e·Å."""
        return ornstein_uhlenbeck(200_000, dt=self.dt, tau=self.tau)

    def debye(self, dipole, nu):
        """Analytic Debye susceptibility for the sampled dipole variance."""
        pref = scipy.constants.e**2 * scipy.constants.angstrom**2
        pref /= 3 * self.volume * scipy.constants.angstrom**3
        pref /= scipy.constants.k * self.temperature * scipy.constants.epsilon_0
        chi_0 = pref * np.sum(np.var(dipole, axis=0))
        return chi_0 / (1 - 2j * np.pi * nu * self.tau)

    def band(self, nu):
        """Mask for 0.5 < ωτ < 2, around the loss peak."""
        omega_tau = 2 * np.pi * nu * self.tau
        return (omega_tau > 0.5) & (omega_tau < 2)

    @pytest.mark.parametrize(("n_dim", "sigma"), [(1, 1.0), (3, 2.5)])
    def test_ornstein_uhlenbeck(self, n_dim, sigma):
        """The reference process has variance σ² and decays as exp(-t/τ).

        Checked with plain numpy, independent of spectrumkit.
        """
        x = ornstein_uhlenbeck(
            200_000, n_dim=n_dim, dt=self.dt, tau=self.tau, sigma=sigma
        )
        assert x.shape == (200_000, n_dim)

        x = x - x.mean(axis=0)
        lags = np.array([0, 50, 100, 200])
        corr = np.array(
            [np.mean(np.sum(x[: len(x) - k] * x[k:], axis=1)) for k in lags]
        )
        assert_allclose(corr[0] / n_dim, sigma**2, rtol=0.1)
        assert_allclose(corr / corr[0], np.exp(-lags * self.dt / self.tau), atol=0.05)

    def test_periodogram_dipole(self, dipole):
        """Periodogram of the dipole reproduces Im(χ)."""
        res = calculate_spectrum_from_dipole(
            dipole, self.dt, self.volume, self.temperature, segs=100, nobin=True
        )
        m = self.band(res["nu"])
        expected = self.debye(dipole, res["nu"][m])
        assert_allclose(np.mean(res["susc"].imag[m] / expected.imag), 1, rtol=0.05)

    def test_periodogram_current(self, dipole):
        """Periodogram of the current reproduces Im(χ)."""
        current = np.diff(dipole, axis=0) / self.dt
        res = calculate_spectrum_from_current(
            current, self.dt, self.volume, self.temperature, segs=100, nobin=True
        )
        m = self.band(res["nu"])
        expected = self.debye(dipole, res["nu"][m])
        assert_allclose(np.mean(res["susc"].imag[m] / expected.imag), 1, rtol=0.05)

    @pytest.mark.parametrize("spectrum_type", ["dipole", "flux"])
    def test_wiener_khinchin(self, dipole, spectrum_type):
        """WienerKhinchin, averaged over blocks, reproduces Im(χ)."""
        series = dipole
        if spectrum_type == "flux":
            series = np.diff(dipole, axis=0) / self.dt
        block = 2000
        time = self.dt * np.arange(block)
        results = [
            WienerKhinchin(
                time,
                series[i : i + block],
                self.volume,
                self.temperature,
                spectrum_type=spectrum_type,
            )
            for i in range(0, len(series) - block + 1, block)
        ]
        nu = results[0][0] / (2 * np.pi)
        susc = np.mean([s for _, s in results], axis=0)
        m = self.band(nu)
        expected = self.debye(dipole, nu[m])
        assert_allclose(np.mean(susc.imag[m] / expected.imag), 1, rtol=0.05)
