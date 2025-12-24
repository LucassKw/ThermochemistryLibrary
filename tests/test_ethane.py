from .ethane_input import freqs, masses, coords, principal_moments, T, linear, mass_kg, elec_energy
from thermochemistry_library.hessian.enthalpy import Enthalpy
from thermochemistry_library.hessian.entropy import Entropy
from thermochemistry_library.hessian.gibbs import Gibbs

def test_ethane_properties():
    enthalpy = Enthalpy(freqs_cm=freqs, T=T, linear=linear, electronic_energy=elec_energy)
    entropy = Entropy(T=T, mass_kg=mass_kg, principal_moments=principal_moments, frequencies_cm=freqs, linear=linear)

    H = enthalpy.total_enthalpy()
    S = entropy.total_entropy(correction_1M=False)
    G = Gibbs(enthalpy, entropy).gibbs_energy()

    print(f"H = {H/1000} kJ/mol")
    print(f"S = {S} J/mol·K")
    print(f"G = {G} kJ/mol")

    import numpy as np
    assert np.isclose(H/1000, -209520.605, atol=1e-1)
    assert np.isclose(S, 246.689, atol=1e-1)
    assert np.isclose(G, -209594.156, atol=1e-1)

if __name__ == "__main__":
    test_ethane_properties()
