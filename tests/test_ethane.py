import sys
import numpy as np

from tests.ethane_data import HESSIAN_RAW, parse_hessian
from thermochemistry_library.hessian.thermo import calculate_thermo
from thermochemistry_library.hessian.vibration import VibrationalAnalysis


T = 298.150
LINEAR = False
elem_energy = -79.8304209466

masses = np.array([
    12.0000000,
    1.0078250,
    1.0078250,
    1.0078250,
    12.0000000,
    1.0078250,
    1.0078250,
    1.0078250
])


coords = np.array([
    [ 0.746629,  0.013229,  0.187809],
    [ 1.338190, -0.136988, -0.691050],
    [ 0.988256,  0.960502,  0.622772],
    [ 0.950960, -0.765432,  0.892675],
    [-0.746627, -0.013228, -0.187807],
    [-1.338230,  0.135059,  0.691346],
    [-0.951353,  0.766656, -0.891208],
    [-0.987829, -0.959802, -0.624549]
])


if __name__ == "__main__":
    hessian = parse_hessian(HESSIAN_RAW)

    vib = VibrationalAnalysis(hessian, masses, coords, hessian_in_angstrom=False)
    vib_results = vib.run()
    freqs = vib_results["frequencies"]

    print(f"Found {len(freqs)} Frequencies (cm^-1):")

    for i, f in enumerate(freqs):
        print(f"{f:10.4f}", end="\n" if (i+1)%3==0 else "  ")
    print("\n") 

    try:
        results = calculate_thermo(
            hessian=hessian,
            masses=masses,
            coords=coords,
            temperature=T,
            electronic_energy=elem_energy,
            linear=LINEAR,
            correction_1m=False,
            symmetry_number=1,
            hessian_in_angstrom=False
        )

        print("\n" + "="*40)
        print(" RESULTS")
        print("="*40)
        print(f"Zero-Point Energy:  {results.zero_point_energy:.6f} kJ/mol")
        print(f"Enthalpy (H):       {results.enthalpy:.6f} kJ/mol")
        print(f"Entropy (S):        {results.entropy:.6f} J/mol·K")
        print(f"Gibbs Energy (G):   {results.gibbs_energy:.6f} kJ/mol")
        print("="*40)

    except Exception as e:
        print(f"\nCalculation Failed: {e}")
        import traceback
        traceback.print_exc()
