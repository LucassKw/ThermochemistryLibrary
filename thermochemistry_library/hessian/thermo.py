"""Thermochemistry helper orchestration."""

from dataclasses import dataclass

import numpy as np

from .enthalpy import Enthalpy
from .entropy import Entropy
from .gibbs import Gibbs
from .vibration import VibrationalAnalysis


@dataclass
class ThermoResults:
    """Dataclass for storing thermochemistry results."""

    enthalpy: float
    entropy: float
    gibbs_energy: float
    zero_point_energy: float


def calculate_thermo(
    hessian: np.ndarray,
    masses: np.ndarray,
    coords: np.ndarray,
    temperature: float,
    linear: bool = False,
    correction_1m: bool = True,
    electronic_energy: float = 0.0,
    frequency_scale_factor: float = 1.0,
    quasi_harmonic: bool = False,
    qh_type: str = "grimme",
    qh_cutoff: float = 100.0,
) -> ThermoResults:
    """Calculate thermochemistry properties."""
    # 1. Run Vibrational Analysis
    vib = VibrationalAnalysis(hessian, masses, coords)
    vib_results = vib.run()

    freqs = vib_results["frequencies"]
    # Apply frequency scaling
    freqs = freqs * frequency_scale_factor

    principal_moments = vib_results["principal"]

    is_linear = vib.is_linear

    # 2. Prepare inputs for Enthalpy/Entropy
    total_mass_amu = np.sum(masses)
    total_mass_kg = total_mass_amu * 1.66053906660e-27

    # 3. Calculate Thermochemistry
    enthalpy_calc = Enthalpy(
        freqs_cm=freqs,
        T=temperature,
        linear=is_linear,
        electronic_energy=electronic_energy,
        quasi_harmonic=quasi_harmonic,
        qh_cutoff=qh_cutoff,
    )

    # Determine entropy QH settings
    # If quasi_harmonic is True, use qh_type. If False, qh_type is None.
    entropy_qh_type = qh_type if quasi_harmonic else None

    entropy_calc = Entropy(
        T=temperature,
        mass_kg=total_mass_kg,
        principal_moments=principal_moments,
        frequencies_cm=freqs,
        linear=is_linear,
        qh_type=entropy_qh_type,
        qh_cutoff=qh_cutoff,
    )
    gibbs_calc = Gibbs(enthalpy_calc, entropy_calc)

    return ThermoResults(
        enthalpy=enthalpy_calc.total_enthalpy() / 1000.0,
        entropy=entropy_calc.total_entropy(correction_1m=correction_1m),
        gibbs_energy=gibbs_calc.gibbs_energy(),
        zero_point_energy=enthalpy_calc.zero_point_energy() / 1000.0,
    )
