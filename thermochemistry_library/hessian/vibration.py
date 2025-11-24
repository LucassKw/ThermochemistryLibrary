import numpy as np
import scipy.linalg
from dataclasses import dataclass, field

C = 2.99792458e10
ANGSTROM_PER_BOHR = 0.529177210903
LIGHTSPEED_SI = 299792458
AU_TO_INVCM = 219474.63

@dataclass
class VibrationalAnalysis:
    hessian_input: np.ndarray
    masses_amu: np.ndarray
    coordinates: np.ndarray
    
    mass_weighted: np.ndarray = field(init=False, default=None)
    eigvals: np.ndarray = field(init=False, default=None)
    eigvecs: np.ndarray = field(init=False, default=None)
    reduced_masses: np.ndarray = field(init=False, default=None)
    frequencies: np.ndarray = field(init=False, default=None)
    force_constants: np.ndarray = field(init=False, default=None)
    principal: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        self.hessian = np.array(self.hessian_input)
        self.masses = np.array(self.masses_amu)
        self.coords = np.array(self.coordinates)
        self.n_atoms = len(self.masses)
        self.n_dir = 3 * self.n_atoms

    @property
    def is_linear(self):
        moments = self.principal_moments()
        if moments[2] < 1e-6:
            return False
        return moments[0] < 1e-4 * moments[2]

    def center_of_mass(self):
        total_mass = np.sum(self.masses)
        com = np.sum(self.masses[:, np.newaxis] * self.coords, axis=0) / total_mass
        return com

    def inertia_tensor(self):
        com = self.center_of_mass()
        shifted = self.coords - com

        I = np.zeros((3, 3))

        for i in range(self.n_atoms):
            m = self.masses[i]
            x, y, z = shifted[i]

            I[0, 0] += m * (y**2 + z**2)
            I[1, 1] += m * (x**2 + z**2)
            I[2, 2] += m * (x**2 + y**2)
            I[0, 1] -= m * x * y
            I[0, 2] -= m * x * z
            I[1, 2] -= m * y * z

        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]

        return I

    def principal_moments(self):
        I = self.inertia_tensor()
        eigvals, _ = np.linalg.eigh(I)
        self.principal = np.sort(eigvals)
        return self.principal

    def _generate_translation_rotation_vectors(self):
        N = self.n_atoms
        
        # translation vectors
        d1 = np.zeros(shape=3 * N)
        d1[::3] = np.sqrt(self.masses)
        d2 = np.roll(d1, shift=1)
        d3 = np.roll(d1, shift=2)

        # principal axes of rotation
        I = self.inertia_tensor()
        _, X = np.linalg.eigh(I)

        # rotation vectors
        d4, d5, d6 = np.zeros(shape=(3 * N)), np.zeros(shape=(3 * N)), np.zeros(shape=(3 * N))
        sqrt_mass = np.sqrt(self.masses)
        positions = self.coords - self.center_of_mass()
        
        for i, (position, sqrt_mass_i) in enumerate(zip(positions, sqrt_mass)):
            P = position @ X
            d4[3 * i : 3 * i + 3] = (P[1] * X[:, 2] - P[2] * X[:, 1]) * sqrt_mass_i
            d5[3 * i : 3 * i + 3] = (P[2] * X[:, 0] - P[0] * X[:, 2]) * sqrt_mass_i
            d6[3 * i : 3 * i + 3] = (P[0] * X[:, 1] - P[1] * X[:, 0]) * sqrt_mass_i

        # check if the vectors are real and normalize
        ds = []
        for d in [d1, d2, d3, d4, d5, d6]:
            norm_sq = np.dot(d, d)
            if norm_sq > 1e-4:
                d *= 1 / np.sqrt(norm_sq)
                ds.append(d)

        return np.stack(ds)

    def run(self):
        N = self.n_atoms
        
        H_cart = self.hessian.copy()
        H_cart *= ANGSTROM_PER_BOHR**2

        # Mass-weighted Hessian
        H_mwc = H_cart.copy()
        
        mass_vec = np.repeat(self.masses, 3)
        inv_sqrt_mass = 1.0 / np.sqrt(mass_vec)
        H_mwc = H_mwc * np.outer(inv_sqrt_mass, inv_sqrt_mass)
        
        self.mass_weighted = H_mwc

        # Get matrix D which eliminates rotational/translational motion
        D_tr = self._generate_translation_rotation_vectors()
        D = scipy.linalg.null_space(D_tr)

        # Convert hessian to new coordinates and find freqs and modes
        H_int = D.T @ H_mwc @ D
        e, L = np.linalg.eigh(H_int)

        # Eigenvalues to frequencies
        CONVERSION_FACTOR = 5140.4847323

        freqs_cm = []
        for val in e:
            if val >= 0:
                freqs_cm.append(np.sqrt(val) * CONVERSION_FACTOR)
            else:
                freqs_cm.append(-np.sqrt(abs(val)) * CONVERSION_FACTOR)
        
        freqs_cm = np.array(freqs_cm)

        # Mass diagonal matrix (M)
        M = np.diag(inv_sqrt_mass)

        # Get modes in cartesian coordinates
        L_cart = M @ D @ L
        
        num_modes = L_cart.shape[1]
        
        self.frequencies = freqs_cm
        self.reduced_masses = np.zeros(num_modes)
        self.force_constants = np.zeros(num_modes)
        self.eigvecs = L_cart 

        # Normalize and get reduced mass and force constants
        for i in range(num_modes):
            # Normalization
            norm_sq = np.dot(L_cart[:, i], L_cart[:, i])
            N_i = np.sqrt(1.0 / norm_sq)
            
            # Reduced mass
            mu = N_i**2
            self.reduced_masses[i] = mu
            
            # Force constant in mDyne/A
            self.force_constants[i] = 5.89182e-7 * (freqs_cm[i]**2) * mu

        return {
            "frequencies": self.frequencies,
            "reduced_masses": self.reduced_masses,
            "force_constants": self.force_constants,
            "eigenvectors": self.eigvecs,
            "principal": self.principal_moments()
        }
