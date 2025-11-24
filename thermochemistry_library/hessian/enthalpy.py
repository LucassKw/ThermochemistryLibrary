import numpy as np
from dataclasses import dataclass

h = 6.62607015e-34
c = 2.99792458e10
kB = 1.380649e-23
R = 8.31446262
NA = 6.02214076e23

@dataclass
class Enthalpy:
    freqs_cm: np.ndarray
    T: float
    linear: bool = False

    def __post_init__(self):
        self.freqs = np.array(self.freqs_cm)

    def zero_point_energy(self):
        return 0.5 * h * c * np.sum(self.freqs) * NA

    def vibrational_energy(self):
        theta = (h * c * self.freqs) / kB
        x = theta / self.T
        Evib = R * np.sum(theta / (np.exp(x) - 1))
        return Evib

    def translational_energy(self):
        return 1.5 * R * self.T

    def rotational_energy(self):
        if self.linear:
            Erot = R * self.T
        else:
            Erot = 1.5 * R * self.T
        return Erot

    def total_enthalpy(self):
        return (self.vibrational_energy() + self.translational_energy() + self.rotational_energy() + R * self.T)
