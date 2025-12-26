"""Vibrational analysis utilities."""

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg

C = 2.99792458e10
ANGSTROM_PER_BOHR = 0.529177210903
LIGHTSPEED_SI = 299792458
AU_TO_INVCM = 219474.63
LINEAR_MOMENT_TOL = 1e-6
LINEAR_RATIO_TOL = 1e-4
TR_VECTOR_NORM_TOL = 1e-4


@dataclass
class VibrationalAnalysis:
    """Class for performing vibrational analysis."""

    hessian_input: np.ndarray
    masses_amu: np.ndarray
    coordinates: np.ndarray

    mass_weighted: np.ndarray | None = field(init=False, default=None)
    eigvals: np.ndarray | None = field(init=False, default=None)
    eigvecs: np.ndarray | None = field(init=False, default=None)
    reduced_masses: np.ndarray | None = field(init=False, default=None)
    frequencies: np.ndarray | None = field(init=False, default=None)
    force_constants: np.ndarray | None = field(init=False, default=None)
    principal: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self):
        """Normalize inputs after initialization."""
        self.hessian = np.array(self.hessian_input)
        self.masses = np.array(self.masses_amu)
        self.coords = np.array(self.coordinates)
        self.n_atoms = len(self.masses)
        self.n_dir = 3 * self.n_atoms

    @property
    def is_linear(self):
        """Check if molecule is linear based on moments of inertia."""
        moments = self.principal_moments()
        if moments[2] < LINEAR_MOMENT_TOL:
            return False
        return moments[0] < LINEAR_RATIO_TOL * moments[2]

    def center_of_mass(self):
        """Calculate center of mass."""
        total_mass = np.sum(self.masses)
        com = np.sum(self.masses[:, np.newaxis] * self.coords, axis=0) / total_mass
        return com

    def inertia_tensor(self):
        """Calculate inertia tensor."""
        com = self.center_of_mass()
        shifted = self.coords - com

        inertia = np.zeros((3, 3))

        for i in range(self.n_atoms):
            m = self.masses[i]
            x, y, z = shifted[i]

            inertia[0, 0] += m * (y**2 + z**2)
            inertia[1, 1] += m * (x**2 + z**2)
            inertia[2, 2] += m * (x**2 + y**2)
            inertia[0, 1] -= m * x * y
            inertia[0, 2] -= m * x * z
            inertia[1, 2] -= m * y * z

        inertia[1, 0] = inertia[0, 1]
        inertia[2, 0] = inertia[0, 2]
        inertia[2, 1] = inertia[1, 2]

        return inertia

    def principal_moments(self):
        """Calculate principal moments of inertia."""
        inertia = self.inertia_tensor()
        eigvals, _ = np.linalg.eigh(inertia)
        self.principal = np.sort(eigvals)
        return self.principal

    def _generate_translation_rotation_vectors(self):
        n_atoms = self.n_atoms

        # translation vectors
        d1 = np.zeros(shape=3 * n_atoms)
        d1[::3] = np.sqrt(self.masses)
        d2 = np.roll(d1, shift=1)
        d3 = np.roll(d1, shift=2)

        # principal axes of rotation
        inertia = self.inertia_tensor()
        _, x_vec = np.linalg.eigh(inertia)

        # rotation vectors
        d4, d5, d6 = (
            np.zeros(shape=(3 * n_atoms)),
            np.zeros(shape=(3 * n_atoms)),
            np.zeros(shape=(3 * n_atoms)),
        )
        sqrt_mass = np.sqrt(self.masses)
        positions = self.coords - self.center_of_mass()

        for i, (position, sqrt_mass_i) in enumerate(zip(positions, sqrt_mass, strict=True)):
            p_vec = position @ x_vec
            d4[3 * i : 3 * i + 3] = (p_vec[1] * x_vec[:, 2] - p_vec[2] * x_vec[:, 1]) * sqrt_mass_i
            d5[3 * i : 3 * i + 3] = (p_vec[2] * x_vec[:, 0] - p_vec[0] * x_vec[:, 2]) * sqrt_mass_i
            d6[3 * i : 3 * i + 3] = (p_vec[0] * x_vec[:, 1] - p_vec[1] * x_vec[:, 0]) * sqrt_mass_i

        # check if the vectors are real and normalize
        ds = []
        for vec in [d1, d2, d3, d4, d5, d6]:
            norm_sq = np.dot(vec, vec)
            if norm_sq > TR_VECTOR_NORM_TOL:
                normalized = vec / np.sqrt(norm_sq)
                ds.append(normalized)

        return np.stack(ds)

    def run(self):
        """Run vibrational analysis."""
        # Hessian in Hartree/Bohr^2 -> convert if needed, but here assuming input is consistent
        # Code actually converts it:
        h_cart = self.hessian.copy()
        h_cart *= ANGSTROM_PER_BOHR**2

        # Mass-weighted Hessian
        h_mwc = h_cart.copy()

        mass_vec = np.repeat(self.masses, 3)
        inv_sqrt_mass = 1.0 / np.sqrt(mass_vec)
        h_mwc = h_mwc * np.outer(inv_sqrt_mass, inv_sqrt_mass)

        self.mass_weighted = h_mwc

        # Get matrix D which eliminates rotational/translational motion
        d_tr = self._generate_translation_rotation_vectors()
        d_matrix = scipy.linalg.null_space(d_tr)

        # Convert hessian to new coordinates and find freqs and modes
        h_int = d_matrix.T @ h_mwc @ d_matrix
        e_val, l_mat = np.linalg.eigh(h_int)

        # Eigenvalues to frequencies
        conversion_factor = 5140.4847323

        freqs_cm = []
        for val in e_val:
            if val >= 0:
                freqs_cm.append(np.sqrt(val) * conversion_factor)
            else:
                freqs_cm.append(-np.sqrt(abs(val)) * conversion_factor)

        freqs_cm = np.array(freqs_cm)

        # Mass diagonal matrix (M)
        m_diag = np.diag(inv_sqrt_mass)

        # Get modes in cartesian coordinates
        l_cart = m_diag @ d_matrix @ l_mat

        num_modes = l_cart.shape[1]

        self.frequencies = freqs_cm
        self.reduced_masses = np.zeros(num_modes)
        self.force_constants = np.zeros(num_modes)
        self.eigvecs = l_cart

        # Normalize and get reduced mass and force constants
        for i in range(num_modes):
            # Normalization
            norm_sq = np.dot(l_cart[:, i], l_cart[:, i])
            n_i = np.sqrt(1.0 / norm_sq)

            # Reduced mass
            mu = n_i**2
            self.reduced_masses[i] = mu

            # Force constant in mDyne/A
            self.force_constants[i] = 5.89182e-7 * (freqs_cm[i] ** 2) * mu

        return {
            "frequencies": self.frequencies,
            "reduced_masses": self.reduced_masses,
            "force_constants": self.force_constants,
            "eigenvectors": self.eigvecs,
            "principal": self.principal_moments(),
        }
