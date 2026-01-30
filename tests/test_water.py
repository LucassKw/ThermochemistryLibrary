"""Water thermochemistry example using parsed Hessian data."""

import numpy as np

from tests.water_data import HESSIAN_RAW, parse_hessian
from thermochemistry_library.hessian.thermo import calculate_thermo
from thermochemistry_library.hessian.vibration import VibrationalAnalysis

T = 298.15
LINEAR = False
elec_energy = -76.02676074
EXPECTED_FREQ_COUNT = 3

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

    print(f"Found {len(freqs)} Frequencies (cm^-1):")
    for i, f in enumerate(freqs):
        print(f"{f:10.4f}", end="\n" if (i + 1) % 3 == 0 else "  ")
    print("\n")

    results = calculate_thermo(
        hessian=hessian,
        masses=masses,
        coords=coords,
        temperature=T,
        electronic_energy=elec_energy,
        linear=LINEAR,
        correction_1m=False,
        symmetry_number=2,
        hessian_in_angstrom=False,
    )

    print("\n" + "=" * 40)
    print(" RESULTS")
    print("=" * 40)
    print(f"Zero-Point Energy:  {results.zero_point_energy:.6f} kJ/mol")
    print(f"Enthalpy (H):       {results.enthalpy:.6f} kJ/mol")
    print(f"Entropy (S):        {results.entropy:.6f} J/mol·K")
    print(f"Gibbs Energy (G):   {results.gibbs_energy:.6f} kJ/mol")
    print("=" * 40)

    return freqs, results


def test_water():
    """Smoke test that water frequencies compute."""
    freqs, _ = _run_water()
    assert len(freqs) == EXPECTED_FREQ_COUNT


if __name__ == "__main__":
    _run_water()
