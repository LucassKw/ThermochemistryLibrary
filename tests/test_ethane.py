"""Ethane thermochemistry example using parsed Hessian data."""

import numpy as np
import pytest

from tests.ethane_data import HESSIAN_RAW, parse_hessian
from thermochemistry_library.hessian.thermo import calculate_thermo
from thermochemistry_library.hessian.vibration import VibrationalAnalysis

T = 298.150
LINEAR = False
ELEM_ENERGY = -79.8304209466
EXPECTED_FREQ_COUNT = 18
EXPECTED_FREQUENCIES = np.array([
    313.8821,
    832.5922,
    832.9321,
    1009.7576,
    1235.9428,
    1236.1436,
    1433.6860,
    1454.4598,
    1531.8682,
    1532.2034,
    1537.4879,
    1538.0757,
    3046.9410,
    3047.8850,
    3098.2482,
    3098.3500,
    3122.6083,
    3122.6867,
])
EXPECTED_ZERO_POINT_ENERGY = 197.537855
EXPECTED_ENTHALPY = -209385.618168
EXPECTED_ENTROPY = 242.235452
EXPECTED_GIBBS_ENERGY = -209457.840667
ABS_TOL = 1e-3

masses = np.array([
    12.0000000,
    1.0078250,
    1.0078250,
    1.0078250,
    12.0000000,
    1.0078250,
    1.0078250,
    1.0078250,
])


coords = np.array([
    [0.746629, 0.013229, 0.187809],
    [1.338190, -0.136988, -0.691050],
    [0.988256, 0.960502, 0.622772],
    [0.950960, -0.765432, 0.892675],
    [-0.746627, -0.013228, -0.187807],
    [-1.338230, 0.135059, 0.691346],
    [-0.951353, 0.766656, -0.891208],
    [-0.987829, -0.959802, -0.624549],
])


def _run_ethane():
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
        symmetry_number=1,
        hessian_in_angstrom=False,
    )

    return freqs, results


def _print_results(freqs, results):
    """Print ethane frequencies and thermodynamic properties."""
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
def ethane_results():
    """Calculate ethane properties once for all tests in this module."""
    return _run_ethane()


def test_ethane_frequencies(ethane_results):
    """Check ethane frequencies against the reference calculation."""
    freqs, _ = ethane_results

    assert len(freqs) == EXPECTED_FREQ_COUNT
    assert np.all(freqs > 0)
    np.testing.assert_allclose(freqs, EXPECTED_FREQUENCIES, rtol=0, atol=ABS_TOL)


def test_ethane_thermodynamics(ethane_results):
    """Check ethane thermodynamic properties against reference values."""
    _, results = ethane_results

    assert results.zero_point_energy == pytest.approx(EXPECTED_ZERO_POINT_ENERGY, abs=ABS_TOL)
    assert results.enthalpy == pytest.approx(EXPECTED_ENTHALPY, abs=ABS_TOL)
    assert results.entropy == pytest.approx(EXPECTED_ENTROPY, abs=ABS_TOL)
    assert results.gibbs_energy == pytest.approx(EXPECTED_GIBBS_ENERGY, abs=ABS_TOL)


if __name__ == "__main__":
    _print_results(*_run_ethane())
