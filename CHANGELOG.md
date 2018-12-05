# CHANGELOG

## 1.1beta – [no release date yet]

(Among many bugfixes...)

### I/O

- More permisive resuming.

### Parameterization and priors

- Made possible to fix a parameter whose only role is being an argument of a dynamically defined one.

### Samplers

#### General

- Seeded runs for `evaluate`, `mcmc` and `polychord`.

### Cosmological theory codes

#### General

- Added P(k) interpolator as an observable (was already available for CAMB, but not documented)

#### classy

- Updated to 2.7.1
- Added P(k) interpolator

### Cosmological likelihoods

#### DES

- Added Y1 release likelihoods [(arXiv:1708.01530)](https://arxiv.org/abs/1708.01530)

#### BICEP-Keck

- Updated to 2015 data [(arXiv:1810.05216)](https://arxiv.org/abs/1810.05216) and renamed to `bicep_keck_2015`.
