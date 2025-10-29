from ethane_input import freqs, masses, coords, principal_moments, T, linear, mass_kg
from hessian.enthalpy import Enthalpy
from hessian.entropy import Entropy
from hessian.gibbs import Gibbs

enthalpy = Enthalpy(freqs_cm=freqs, Temp=T, linear=linear)
entropy = Entropy(T=T, mass_kg=mass_kg, principal_moments=principal_moments, frequencies_cm=freqs, linear=linear)

H = enthalpy.total_enthalpy()
S = entropy.total_entropy(correction_1M=False)
G = Gibbs(enthalpy, entropy).gibbs_energy()

print("H =", H/1000, "kJ/mol")
print("S =", S, "J/mol·K")
print("G =", G, "kJ/mol")
