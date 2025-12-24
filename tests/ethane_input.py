import numpy as np
from thermochemistry_library.hessian.enthalpy import Enthalpy
from thermochemistry_library.hessian.entropy import Entropy
from thermochemistry_library.hessian.gibbs import Gibbs

T = 298.15
linear = False
sigma = 6

mass_amu = 30.06904
mass_kg = mass_amu * 1.66053906660e-27

coords = np.array([
    [ 0.000000,   0.000000,   0.000000],
    [ 1.529000,   0.000000,   0.000000],
    [-0.540000,   0.935000,   0.935000],
    [-0.540000,  -0.935000,  -0.935000],
    [ 2.069000,   0.935000,   0.935000],
    [ 2.069000,  -0.935000,  -0.935000],
    [-0.540000,   0.935000,  -0.935000],
    [-0.540000,  -0.935000,   0.935000],
])

masses = np.array([
    12.00000, 12.00000,
    1.00784, 1.00784,
    1.00784, 1.00784,
    1.00784, 1.00784
])

principal_moments = np.array([
    26.666, 49.734, 49.734
])

freqs = np.array([
     995, 1171, 1380, 1468,
    2898, 2969, 2974, 3006,
    3055, 3071
])

elec_energy = -79.858399 # Hartrees

hessian = None # using frequencies


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
    "electronic_energy": elec_energy,
}
