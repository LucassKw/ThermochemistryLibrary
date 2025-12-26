"""Smoke tests for ethane thermochemistry properties."""

from thermochemistry_library.hessian.enthalpy import Enthalpy
from thermochemistry_library.hessian.entropy import Entropy
from thermochemistry_library.hessian.gibbs import Gibbs

from .ethane_input import T, elec_energy, freqs, linear, mass_kg, principal_moments


def test_ethane_properties():
    """Compute ethane thermochemistry to validate core calculations."""
    enthalpy = Enthalpy(freqs_cm=freqs, T=T, linear=linear, electronic_energy=elec_energy)
    entropy = Entropy(
        T=T,
        mass_kg=mass_kg,
        principal_moments=principal_moments,
        frequencies_cm=freqs,
        linear=linear,
    )

    H = enthalpy.total_enthalpy()
    S = entropy.total_entropy(correction_1m=False)
    G = Gibbs(enthalpy, entropy).gibbs_energy()

    print(f"H = {H / 1000} kJ/mol")
    print(f"S = {S} J/mol·K")
    print(f"G = {G} kJ/mol")


if __name__ == "__main__":
    test_ethane_properties()
