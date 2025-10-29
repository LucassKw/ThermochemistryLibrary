import numpy as np
from .enthalpy import Enthalpy
from .entropy import Entropy

h = 6.62607015e-34
c = 2.99792458e10
R = 8.314462618
NA = 6.02214076e23

class Gibbs:

    def __init__(self, enthalpy: Enthalpy, entropy: Entropy):
        self.enthalpy = enthalpy
        self.entropy = entropy
        self.T = enthalpy.T


    def zero_point_energy(self):

        return self.enthalpy.zero_point_energy()


    def gibbs_energy(self):

        H = self.enthalpy.total_enthalpy()
        S = self.entropy.total_entropy(False)

        return (H - self.T * S)/1000.0