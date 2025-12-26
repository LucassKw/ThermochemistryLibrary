# Thermochemistry Library

[![License](https://img.shields.io/github/license/LucassKw/ThermochemistryLibrary)](
https://github.com/LucassKw/ThermochemistryLibrary/blob/main/LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](
https://github.com/astral-sh/ruff)

This library converts a cartesian hessian of nuclear second derivatives into vibrational and thermodynamic quantities, following similar framework used in Gaussian and [GoodVibes](https://github.com/patonlab/GoodVibes), but in a modular and Python-native implementation.

#### Minimum Inputs

```
• Cartesian Hessian Matrix
• Atomic Masses
• Cartesian Coordinates
• Temperature (K)
```

#### Example Outputs

```
• Vibrational Frequencies
• Normal Modes
• Cartesian Coordinates
• Rotational Constants
• Center of Mass
• Entropy
• Enthalpy
• Gibbs Free Energy
```

#### Quasi-Harmonic Corrections

Akin to goodvibes, the [Cramer & Truhlar](https://pubs.acs.org/doi/abs/10.1021/jp205508z) and [Grimme](https://chemistry-europe.onlinelibrary.wiley.com/doi/abs/10.1002/chem.201200497) (RRHO) approximations were used. 

```
Cramer & Truhlar:
mod_freqs = np.where(mod_freqs < self.qh_cutoff, self.qh_cutoff, mod_freqs)

(qh_cutoff = 100cm^-1)
```

```
Grimme:
w = 1.0 / (1.0 + (self.qh_cutoff / valid_freqs)**4)
```

## Credits
This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/uv-cookiecutter](https://github.com/jevandezande/uv-cookiecutter) project template.

This library was in loose collabration with [Rowansci](https://rowansci.com/)
