"""Enthalpy calculations for vibrational thermochemistry."""

from dataclasses import dataclass

import numpy as np

h = 6.62607015e-34
c = 2.99792458e10
K_B = 1.380649e-23
R = 8.31446262
NA = 6.02214076e23
HARTREE_TO_JOULES = 4.3597447222071e-18


@dataclass
class Enthalpy:
    """Class for calculating Enthalpy."""

    freqs_cm: np.ndarray
    T: float
    linear: bool = False
    electronic_energy: float = 0.0  # in Hartrees
    quasi_harmonic: bool = False
    qh_cutoff: float = 100.0

    def __post_init__(self):
        """Normalize inputs after initialization."""
        self.freqs = np.array(self.freqs_cm)

    def zero_point_energy(self):
        """Calculate zero point energy."""
        return 0.5 * h * c * np.sum(self.freqs) * NA

    def _calc_damp(self):
        # Damping function w(v) = 1 / (1 + (v0/v)^4)
        valid_freqs = self.freqs[self.freqs > 0]
        w = 1.0 / (1.0 + (self.qh_cutoff / valid_freqs) ** 4)
        return w

    def vibrational_energy(self):
        """Calculate vibrational energy."""
        valid_freqs = self.freqs[self.freqs > 0]
        theta = (h * c * valid_freqs) / K_B
        x = theta / self.T
        evib_rrho_modes = R * theta / (np.exp(x) - 1)

        if self.quasi_harmonic:
            w = self._calc_damp()
            # Blend between RRHO and 0.5 RT
            evib_modes = w * evib_rrho_modes + (1 - w) * (0.5 * R * self.T)
            return np.sum(evib_modes)
        else:
            return np.sum(evib_rrho_modes)

    def translational_energy(self):
        """Calculate translational energy."""
        return 1.5 * R * self.T

    def rotational_energy(self):
        """Calculate rotational energy."""
        if self.linear:
            erot = R * self.T
        else:
            erot = 1.5 * R * self.T
        return erot

    def total_enthalpy(self):
        """Return total enthalpy in J/mol."""
        h_thermal = (
            self.vibrational_energy()
            + self.translational_energy()
            + self.rotational_energy()
            + R * self.T
        )

        if self.electronic_energy != 0.0:
            e_elec_j_mol = self.electronic_energy * HARTREE_TO_JOULES * NA
            return e_elec_j_mol + self.zero_point_energy() + h_thermal

        return h_thermal
