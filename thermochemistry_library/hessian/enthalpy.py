from dataclasses import dataclass

import numpy as np

h = 6.62607015e-34
c = 2.99792458e10
kB = 1.380649e-23
R = 8.31446262
NA = 6.02214076e23
HARTREE_TO_JOULES = 4.3597447222071e-18

@dataclass
class Enthalpy:
    freqs_cm: np.ndarray
    T: float
    linear: bool = False
    electronic_energy: float = 0.0  # in Hartrees
    quasi_harmonic: bool = False
    qh_cutoff: float = 100.0

    def __post_init__(self):
        self.freqs = np.array(self.freqs_cm)

    def zero_point_energy(self):
        return 0.5 * h * c * np.sum(self.freqs) * NA

    def _calc_damp(self):
        # Damping function w(v) = 1 / (1 + (v0/v)^4)
        valid_freqs = self.freqs[self.freqs > 0]
        w = 1.0 / (1.0 + (self.qh_cutoff / valid_freqs)**4)
        return w

    def vibrational_energy(self):
        valid_freqs = self.freqs[self.freqs > 0]
        theta = (h * c * valid_freqs) / kB
        x = theta / self.T
        Evib_rrho_modes = R * theta / (np.exp(x) - 1)
        
        if self.quasi_harmonic:
            w = self._calc_damp()
            # Blend between RRHO and 0.5 RT
            Evib_modes = w * Evib_rrho_modes + (1 - w) * (0.5 * R * self.T)
            return np.sum(Evib_modes)
        else:
            return np.sum(Evib_rrho_modes)

    def translational_energy(self):
        return 1.5 * R * self.T

    def rotational_energy(self):
        if self.linear:
            Erot = R * self.T
        else:
            Erot = 1.5 * R * self.T
        return Erot

    def total_enthalpy(self):
        """
        Returns total enthalpy in J/mol.
        If electronic_energy is provided (non-zero), it adds Electronic Energy + ZPE to the thermal terms.
        """
        H_thermal = self.vibrational_energy() + self.translational_energy() + self.rotational_energy() + R * self.T

        if self.electronic_energy != 0.0:
             E_elec_J_mol = self.electronic_energy * HARTREE_TO_JOULES * NA
             return E_elec_J_mol + self.zero_point_energy() + H_thermal

        return H_thermal
