"""Water thermochemistry example using parsed Hessian data."""

import numpy as np
import pytest

from tests.water_data import HESSIAN_RAW, parse_hessian
from thermochemistry_library.hessian.thermo import calculate_thermo
from thermochemistry_library.hessian.vibration import VibrationalAnalysis

T = 298.15
LINEAR = False
ELEM_ENERGY = -76.3681281356
EXPECTED_FREQ_COUNT = 3
EXPECTED_FREQUENCIES = np.array([1694.8302, 3644.5602, 3778.7036])
EXPECTED_ZERO_POINT_ENERGY = 54.538314
EXPECTED_ENTHALPY = -200440.033055
EXPECTED_ENTROPY = 188.702164
EXPECTED_GIBBS_ENERGY = -200496.294605
ABS_TOL = 1e-3

masses = np.array([
    15.999,
    1.0078,
    1.0078,
])

coords = np.array([
    [0.000000, 0.000000, 0.117790],
    [0.000000, 0.757160, -0.471161],
    [0.000000, -0.757160, -0.471161],
])


def _run_water():
    hessian = parse_hessian(HESSIAN_RAW)
    vib = VibrationalAnalysis(hessian, masses, coords, hessian_in_angstrom=False)
    vib_results = vib.run()
    freqs = vib_results["frequencies"]

    results = calculate_thermo(
        hessian=hessian,
        masses=masses,
        coords=coords,
        temperature=T,
        electronic_energy=ELEM_ENERGY,
        linear=LINEAR,
        correction_1m=False,
        symmetry_number=2,
        hessian_in_angstrom=False,
    )

    return freqs, results


def _print_results(freqs, results):
    """Print water frequencies and thermodynamic properties."""
    print(f"Found {len(freqs)} Frequencies (cm^-1):")
    for i, frequency in enumerate(freqs):
        print(f"{frequency:10.4f}", end="\n" if (i + 1) % 3 == 0 else "  ")
    print("\n")
    print("=" * 40)
    print(" RESULTS")
    print("=" * 40)
    print(f"Zero-Point Energy:  {results.zero_point_energy:.6f} kJ/mol")
    print(f"Enthalpy (H):       {results.enthalpy:.6f} kJ/mol")
    print(f"Entropy (S):        {results.entropy:.6f} J/mol·K")
    print(f"Gibbs Energy (G):   {results.gibbs_energy:.6f} kJ/mol")
    print("=" * 40)


@pytest.fixture(scope="module")
def water_results():
    return _run_water()


def test_water_frequencies(water_results):
    freqs, _ = water_results

    assert len(freqs) == EXPECTED_FREQ_COUNT
    assert np.all(freqs > 0)
    np.testing.assert_allclose(freqs, EXPECTED_FREQUENCIES, rtol=0, atol=ABS_TOL)


def test_water_thermodynamics(water_results):
    _, results = water_results

    assert results.zero_point_energy == pytest.approx(EXPECTED_ZERO_POINT_ENERGY, abs=ABS_TOL)
    assert results.enthalpy == pytest.approx(EXPECTED_ENTHALPY, abs=ABS_TOL)
    assert results.entropy == pytest.approx(EXPECTED_ENTROPY, abs=ABS_TOL)
    assert results.gibbs_energy == pytest.approx(EXPECTED_GIBBS_ENERGY, abs=ABS_TOL)


if __name__ == "__main__":
    _print_results(*_run_water())
