from dataclasses import dataclass
from .enthalpy import Enthalpy
from .entropy import Entropy

h = 6.62607015e-34
c = 2.99792458e10
R = 8.314462618
NA = 6.02214076e23

@dataclass
class Gibbs:
    enthalpy: Enthalpy
    entropy: Entropy

    def __post_init__(self):
        self.T = self.enthalpy.T

    def zero_point_energy(self):
        return self.enthalpy.zero_point_energy()

    def gibbs_energy(self):
        H = self.enthalpy.total_enthalpy()
        S = self.entropy.total_entropy(False)
        return (H - self.T * S)/1000.0