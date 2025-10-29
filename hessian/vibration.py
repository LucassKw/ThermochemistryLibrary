import numpy as np

C = 2.99792458e10

class VibrationalAnalysis:

    def __init__(self, hessian, masses_amu, coordinates):

        self.hessian = np.array(hessian)
        self.masses = np.array(masses_amu)
        self.coords = np.array(coordinates)
        self.n_atoms = len(self.masses)
        self.n_dir = 3 * self.n_atoms

        self.mass_weighted = None
        self.eigvals = None
        self.eigvecs = None
        self.reduced_masses = None
        self.frequencies = None
        self.force_constants = None
        self.principle = None


    def mass_weight_hessian(self):

        mwf = self.hessian.copy()

        for i in range(self.n_dir):

            for j in range(self.n_dir):

                mi = self.masses[i // 3]
                mj = self.masses[j // 3]
                mwf[i, j] /= np.sqrt(mi * mj)

        self.mass_weighted = mwf

        return mwf


    def diagonalize(self):

        eigvals, eigvecs = np.linalg.eigh(self.mass_weighted)

        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)

        self.eigvals = eigvals
        self.eigvecs = eigvecs

        return eigvals, eigvecs


    def compute_reduced_masses(self):

        rm = np.zeros(self.n_dir)

        for i in range(self.n_dir):

            square = 0.0

            for k in range(self.n_dir):

                atom_index = k // 3
                square += (self.eigvecs[k, i] ** 2) / self.masses[atom_index]

            rm[i] = 1.0 / square

        self.reduced_masses = rm
        
        return rm


    def compute_frequencies(self):

        freqs = np.zeros(self.n_dir)

        for i, lambdbd in enumerate(self.eigvals):

            if lambd < 0:
                freqs[i] = -np.sqrt(abs(lambd)) / (2 * np.pi * C)
            else:
                freqs[i] = np.sqrt(lambd) / (2 * np.pi * C)

        self.frequencies = freqs

        return freqs


    def compute_force_constants(self):

        freqs = np.array(self.frequencies)
        rms = np.array(self.reduced_masses)

        k = 4 * (np.pi ** 2) * (C ** 2) * (freqs ** 2) * rms

        self.force_constants = k

        return k
    

    def center_of_mass(self):

        total_mass = np.sum(self.masses)
        com = np.sum(self.masses[:, np.newaxis] * self.coords, axis=0) / total_mass

        return com


    def inertia_axis(self):

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
        
        I = self.inertia_axis()
        eigvals, _ = np.linalg.eigh(I)

        self.principal = np.sort(eigvals)

        return np.sort(eigvals)


    def run(self):

        self.mass_weight_hessian()
        self.diagonalize()
        self.compute_reduced_masses()
        self.compute_frequencies()
        self.compute_force_constants()
        self.principal_moments()

        return {
            "eigenvalues": self.eigvals,
            "eigenvectors": self.eigvecs,
            "reduced_masses": self.reduced_masses,
            "frequencies": self.frequencies,
            "force_constants": self.force_constants,
            "principal": self.principal,
        }


