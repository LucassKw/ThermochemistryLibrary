import numpy as np
from dataclasses import dataclass

c = 2.99792458e10
R = 8.314462618
h = 6.62607015e-34 
kB = 1.380649e-23
NA = 6.02214076e23
p0 = 1.0e5

@dataclass
class Entropy:
    T: float
    mass_kg: float
    principal_moments: np.ndarray
    frequencies_cm: np.ndarray
    linear: bool = False

    def __post_init__(self):
        self.m = self.mass_kg
        self.Ia, self.Ib, self.Ic = np.array(self.principal_moments)
        self.freqs = np.array(self.frequencies_cm)
        self.sigma = 1
        self.g_e = 1

    def translational_entropy(self):
        term1 = ((2 * np.pi * self.m * kB * self.T) / (h**2)) ** 1.5
        term2 = (kB * self.T) / p0
        Strans = R * (np.log(term1 * term2) + 2.5)
        return Strans

    def rotational_entropy(self):
        if self.linear:
            I = ((self.Ib + self.Ic) / 2) * 1.66053907e-47
            Srot = R * (np.log((self.T / self.sigma) * (8 * np.pi**2 * kB * I / h**2)) + 1)
        else:
            Ia, Ib, Ic = (np.array([self.Ia, self.Ib, self.Ic]) * 1.66053907e-47)
            Srot = R * (np.log((np.sqrt(np.pi) / self.sigma) * ((8 * np.pi**2 * kB * self.T / h**2) ** 1.5) * np.sqrt(Ia * Ib * Ic)) + 1.5)
        return Srot
    
    def vibrational_entropy(self):
        theta = (h * c * self.freqs) / kB
        x = theta / self.T
        term1 = x / np.expm1(x)
        term2 = -np.log(1 - np.exp(-x))
        Svib_mode = term1 + term2
        return R * np.sum(Svib_mode)

    def total_entropy(self, correction_1M=True):
        S = (self.translational_entropy() + self.rotational_entropy() + self.vibrational_entropy())
        if correction_1M:
            S += R * np.log(24.46)
        return S