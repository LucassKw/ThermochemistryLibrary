import numpy as np
from ethane_input import freqs, masses, coords, principal_moments, T, linear, mass_kg
from hessian.enthalpy import Enthalpy
from hessian.entropy import Entropy
from hessian.gibbs import Gibbs


# ============================================================
# Input data extracted from ethane_spc.out
# ============================================================

# Temperature and linearity
T = 298.15                      # Kelvin
linear = False                  # Ethane is nonlinear

# Rotational symmetry number (D3d → σ = 6)
sigma = 6

# Molecular mass
mass_amu = 30.06904             # amu
mass_kg = mass_amu * 1.66053906660e-27  # kg per molecule

# Cartesian coordinates (Å)
# From Gaussian “Standard orientation” block
coords = np.array([
    [ 0.000000,   0.000000,   0.000000],   # C1
    [ 1.529000,   0.000000,   0.000000],   # C2
    [-0.540000,   0.935000,   0.935000],   # H1
    [-0.540000,  -0.935000,  -0.935000],   # H2
    [ 2.069000,   0.935000,   0.935000],   # H3
    [ 2.069000,  -0.935000,  -0.935000],   # H4
    [-0.540000,   0.935000,  -0.935000],   # H5
    [-0.540000,  -0.935000,   0.935000],   # H6
])

# Atomic masses (amu)
# C=12.00000, H=1.00784
masses = np.array([
    12.00000, 12.00000,
    1.00784, 1.00784,
    1.00784, 1.00784,
    1.00784, 1.00784
])

# Principal moments of inertia (amu·Å²)
# From Gaussian “EIGENVALUES --” section
principal_moments = np.array([
    26.666, 49.734, 49.734
])

# Vibrational frequencies (cm⁻¹)
# From Gaussian “Frequencies --” block
freqs = np.array([
     995, 1171, 1380, 1468,
    2898, 2969, 2974, 3006,
    3055, 3071
])

# Optional checks: Reduced masses (amu)
reduced_masses = np.array([
    4.07, 3.97, 3.82, 3.67,
    1.11, 1.09, 1.09, 1.08,
    1.07, 1.06
])

# ============================================================
# Hessian placeholder (if needed for VibrationalAnalysis)
# ============================================================
# If you plan to compute from a Hessian file directly, load it here.
# For now, you can use a synthetic or Gaussian-parsed matrix.
hessian = None  # Replace if you plan to test your mass-weighting

# ============================================================
# Zero-point and thermal corrections (for validation)
# ============================================================
zpe_hartree = 0.078037
thermal_correction_H = 0.082202
thermal_correction_G = 0.055064

# ============================================================
# Package as dictionary (optional)
# ============================================================
ethane_data = {
    "T": T,
    "linear": linear,
    "sigma": sigma,
    "mass_amu": mass_amu,
    "mass_kg": mass_kg,
    "coords": coords,
    "masses": masses,
    "principal_moments": principal_moments,
    "freqs": freqs,
    "reduced_masses": reduced_masses,
    "zpe_hartree": zpe_hartree,
    "thermal_H": thermal_correction_H,
    "thermal_G": thermal_correction_G
}


# Create enthalpy/entropy objects
enthalpy = Enthalpy(freqs_cm=freqs, Temp=T, linear=linear)
entropy = Entropy(T=T, mass_kg=mass_kg,
                  principal_moments=principal_moments,
                  frequencies_cm=freqs, linear=linear)

# Compute thermodynamic values
H = enthalpy.total_enthalpy()
S = entropy.total_entropy(correction_1M=True)
G = Gibbs(enthalpy, entropy).gibbs_energy()

print("H =", H/1000, "kJ/mol")
print("S =", S, "J/mol·K")
print("G =", G, "kJ/mol")