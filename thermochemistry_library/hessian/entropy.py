"""Entropy calculations for vibrational thermochemistry."""

from dataclasses import dataclass

import numpy as np

c = 2.99792458e10
R = 8.314462618
h = 6.62607015e-34
K_B = 1.380649e-23
NA = 6.02214076e23
p0 = 1.0e5


@dataclass
class Entropy:
    """Class for calculating vibrational, rotational, and translational entropy."""

    T: float
    mass_kg: float
    principal_moments: np.ndarray
    frequencies_cm: np.ndarray
    linear: bool = False
    qh_type: str | None = None
    qh_cutoff: float = 100.0

    def __post_init__(self):
        """Normalize inputs after initialization."""
        self.m = self.mass_kg
        self.Ia, self.Ib, self.Ic = np.array(self.principal_moments)
        self.freqs = np.array(self.frequencies_cm)
        self.sigma = 1
        self.g_e = 1
        self.BAV = 1.00e-44  # Average moment of inertia for Grimme's method

    def translational_entropy(self):
        """Calculate translational entropy using the Sackur-Tetrode equation."""
        term1 = ((2 * np.pi * self.m * K_B * self.T) / (h**2)) ** 1.5
        term2 = (K_B * self.T) / p0
        s_trans = R * (np.log(term1 * term2) + 2.5)
        return s_trans

    def rotational_entropy(self):
        """Calculate rotational entropy based on moments of inertia."""
        if self.linear:
            inertia = ((self.Ib + self.Ic) / 2) * 1.66053907e-47
            s_rot = R * (
                np.log((self.T / self.sigma) * (8 * np.pi**2 * K_B * inertia / h**2)) + 1
            )
        else:
            ia, ib, ic = np.array([self.Ia, self.Ib, self.Ic]) * 1.66053907e-47
            s_rot = R * (
                np.log(
                    (np.sqrt(np.pi) / self.sigma)
                    * ((8 * np.pi**2 * K_B * self.T / h**2) ** 1.5)
                    * np.sqrt(ia * ib * ic)
                )
                + 1.5
            )
        return s_rot

    def _calc_rrho_entropy(self, freqs):
        # Calculate harmonic entropy for an array of frequencies
        # Filter out negative/imaginary frequencies just in case.
        # These should be handled before this step.
        valid_freqs = freqs[freqs > 0]
        theta = (h * c * valid_freqs) / K_B
        x = theta / self.T
        term1 = x / np.expm1(x)
        term2 = -np.log(1 - np.exp(-x))
        return R * np.sum(term1 + term2)

    def _calc_free_rotor_entropy(self):
        # Calculate free rotor entropy for each mode (Grimme)
        # mu = h / (8 * pi^2 * v * c)
        valid_freqs = self.freqs[self.freqs > 0]
        mu = h / (8 * np.pi**2 * valid_freqs * c)
        mu_prime = (mu * self.BAV) / (mu + self.BAV)

        factor = (8 * np.pi**3 * mu_prime * K_B * self.T) / (h**2)
        s_rotor = R * (0.5 + np.log(np.sqrt(factor)))
        return s_rotor

    def _calc_damp(self):
        # Damping function w(v) = 1 / (1 + (v0/v)^4)
        valid_freqs = self.freqs[self.freqs > 0]
        w = 1.0 / (1.0 + (self.qh_cutoff / valid_freqs) ** 4)
        return w

    def vibrational_entropy(self):
        """Calculate vibrational entropy with optional quasi-harmonic corrections."""
        if self.qh_type is None:
            return self._calc_rrho_entropy(self.freqs)

        if self.qh_type.lower() == "grimme":
            # S = w * S_RRHO + (1-w) * S_rotor
            # we need per-mode entropy for mixing
            valid_freqs = self.freqs[self.freqs > 0]

            # RRHO per mode
            theta = (h * c * valid_freqs) / K_B
            x = theta / self.T
            s_rrho_modes = R * (x / np.expm1(x) - np.log(1 - np.exp(-x)))

            # Free rotor per mode
            s_rotor_modes = self._calc_free_rotor_entropy()

            # Damping
            w = self._calc_damp()

            s_vib = np.sum(w * s_rrho_modes + (1 - w) * s_rotor_modes)
            return s_vib

        if self.qh_type.lower() == "truhlar":
            # If freq < cutoff, replace with cutoff for entropy calc
            # effectively: calculate entropy using max(freq, cutoff)

            mod_freqs = np.array(self.freqs)
            mod_freqs = np.where(mod_freqs < self.qh_cutoff, self.qh_cutoff, mod_freqs)
            return self._calc_rrho_entropy(mod_freqs)

        # Fallback to harmonic
        return self._calc_rrho_entropy(self.freqs)

    def total_entropy(self, correction_1m=True):
        """Calculate total entropy.

        Parameters
        ----------
        correction_1m : bool
            Whether to apply the standard state correction (1 atm -> 1 M).
        """
        s_total = (
            self.translational_entropy() + self.rotational_entropy() + self.vibrational_entropy()
        )
        if correction_1m:
            s_total += R * np.log(24.46)
        return s_total
