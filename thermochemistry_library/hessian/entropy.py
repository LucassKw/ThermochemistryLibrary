from dataclasses import dataclass

import numpy as np

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
    qh_type: str = None
    qh_cutoff: float = 100.0

    def __post_init__(self):
        self.m = self.mass_kg
        self.Ia, self.Ib, self.Ic = np.array(self.principal_moments)
        self.freqs = np.array(self.frequencies_cm)
        self.sigma = 1
        self.g_e = 1
        self.BAV = 1.00e-44  # Average moment of inertia for Grimme's method

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

    def _calc_rrho_entropy(self, freqs):
        # Calculate harmonic entropy for an array of frequencies
        # Filter out negative/imaginary frequencies just in case, though they should be handled before
        valid_freqs = freqs[freqs > 0]
        theta = (h * c * valid_freqs) / kB
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
        
        factor = (8 * np.pi**3 * mu_prime * kB * self.T) / (h**2)
        S_rotor = R * (0.5 + np.log(np.sqrt(factor)))
        return S_rotor

    def _calc_damp(self):
        # Damping function w(v) = 1 / (1 + (v0/v)^4)
        valid_freqs = self.freqs[self.freqs > 0]
        w = 1.0 / (1.0 + (self.qh_cutoff / valid_freqs)**4)
        return w

    def vibrational_entropy(self):
        if self.qh_type is None:
            return self._calc_rrho_entropy(self.freqs)
        
        elif self.qh_type.lower() == 'grimme':
            # S = w * S_RRHO + (1-w) * S_rotor
            # we need per-mode entropy for mixing
            valid_freqs = self.freqs[self.freqs > 0]
            
            # RRHO per mode
            theta = (h * c * valid_freqs) / kB
            x = theta / self.T
            S_rrho_modes = R * (x / np.expm1(x) - np.log(1 - np.exp(-x)))
            
            # Free rotor per mode
            S_rotor_modes = self._calc_free_rotor_entropy()
            
            # Damping
            w = self._calc_damp()
            
            S_vib = np.sum(w * S_rrho_modes + (1 - w) * S_rotor_modes)
            return S_vib
            
        elif self.qh_type.lower() == 'truhlar':
            # If freq < cutoff, replace with cutoff for entropy calc
            # effectively: calculate entropy using max(freq, cutoff)
            # GoodVibes logic: 
            # if s_freq_cutoff > 0.0:
            #   if frequency_wn[j] > s_freq_cutoff: vib_entropy.append(Svib_rrho[j])
            #   else: vib_entropy.append(Svib_rrqho[j]) -> which is eval at cutoff
            
            mod_freqs = np.array(self.freqs)
            mod_freqs = np.where(mod_freqs < self.qh_cutoff, self.qh_cutoff, mod_freqs)
            return self._calc_rrho_entropy(mod_freqs)
            
        else:
            # Fallback to harmonic
            return self._calc_rrho_entropy(self.freqs)

    def total_entropy(self, correction_1M=True):
        S = (self.translational_entropy() + self.rotational_entropy() + self.vibrational_entropy())
        if correction_1M:
            S += R * np.log(24.46)
        return S
