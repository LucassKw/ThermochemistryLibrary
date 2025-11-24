import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from thermochemistry_library.hessian.thermo import calculate_thermo, ThermoResults
from .ethane_input import freqs, masses, coords, principal_moments, T, linear

def test_calculate_thermo_mocked():
    with patch('thermochemistry_library.hessian.thermo.VibrationalAnalysis') as MockVib:
        mock_instance = MockVib.return_value
        
        mock_instance.run.return_value = {
            "frequencies": freqs,
            "principal": principal_moments,
            "reduced_masses": np.zeros_like(freqs),

            "force_constants": np.zeros_like(freqs),
            "eigenvectors": np.zeros((len(masses)*3, len(freqs)))
        }
        
        type(mock_instance).is_linear = linear
        
        dummy_hessian = np.zeros((len(masses)*3, len(masses)*3))
        
        results = calculate_thermo(
            hessian=dummy_hessian,
            masses=masses,
            coords=coords,
            T=T,
            correction_1M=False
        )
        
        print(f"Calculated H: {results.enthalpy} kJ/mol")
        print(f"Calculated S: {results.entropy} J/mol/K")
        print(f"Calculated G: {results.gibbs_energy} kJ/mol")
        
        assert isinstance(results, ThermoResults)
        assert np.isclose(results.enthalpy, 10.099889, atol=1e-5)
        assert np.isclose(results.entropy, 246.680333, atol=1e-5)
        assert np.isclose(results.gibbs_energy, -63.447852, atol=1e-5)
