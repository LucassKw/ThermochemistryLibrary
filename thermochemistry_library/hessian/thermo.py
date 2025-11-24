from dataclasses import dataclass
import numpy as np
from .enthalpy import Enthalpy
from .entropy import Entropy
from .gibbs import Gibbs
from .vibration import VibrationalAnalysis

@dataclass
class ThermoResults:
    enthalpy: float
    entropy: float
    gibbs_energy: float
    zero_point_energy: float

def calculate_thermo(
    hessian: np.ndarray,
    masses: np.ndarray,
    coords: np.ndarray,
    T: float,
    linear: bool = False,
    correction_1M: bool = True
) -> ThermoResults:
    
    # 1. Run Vibrational Analysis
    vib = VibrationalAnalysis(hessian, masses, coords)
    vib_results = vib.run()
    
    freqs = vib_results["frequencies"]
    principal_moments = vib_results["principal"]
    
    is_linear = vib.is_linear
    
    # 2. Prepare inputs for Enthalpy/Entropy
    total_mass_amu = np.sum(masses)
    total_mass_kg = total_mass_amu * 1.66053906660e-27
    
    # 3. Calculate Thermochemistry
    enthalpy_calc = Enthalpy(freqs_cm=freqs, T=T, linear=is_linear)
    entropy_calc = Entropy(
        T=T, 
        mass_kg=total_mass_kg, 
        principal_moments=principal_moments, 
        frequencies_cm=freqs, 
        linear=is_linear
    )
    gibbs_calc = Gibbs(enthalpy_calc, entropy_calc)
    
    return ThermoResults(
        enthalpy=enthalpy_calc.total_enthalpy() / 1000.0,
        entropy=entropy_calc.total_entropy(correction_1M=correction_1M),
        gibbs_energy=gibbs_calc.gibbs_energy(),
        zero_point_energy=enthalpy_calc.zero_point_energy() / 1000.0
    )
