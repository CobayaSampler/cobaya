## 2.X  – 2019-XX-XX

### Samplers

#### MCMC

- Added progress tracking (incl. acceptance rate), and a plotting tool for it.
## 2.0.3 – 2019-09-09

### Samplers

#### PolyChord

- Fixed too much oversampling when manual bloking (#35). Thanks Lukas Hergt (@lukashergt), Vivian Miranda (@vivianmiranda) and Will Handley (@williamjameshandley)
- Fixed ifort compatibility (#39, PR #42). Thanks Lukas Hergt (@lukashergt)

#### MCMC

- Fixed: using deprecated Pandas DataFrame method (#40). Thanks Zack Li (@xzackli)

#### Minimize

- Added GetDist output for best-fit (`ignore-prior: True`)

### Likelihoods

- Added `stop_at_error` for likelihoods -- fixes #43. Thanks Lukas Hergt (@lukashergt)

### Cosmology

- Fixed `cobaya-cosmo-generator` in for Python 2 (#37, thanks Lukas Hergt, @lukashergt) and high-DPI screens (#41).


## 2.0 – 2019-08-20

### General

- Added fuzzy matching for names of modules and parameters in a few places. Now error messages show possible misspellings.
- Modules can now be nested, e.g. `planck_2018_lowl.TT` and `planck_2018_lowl.EE` as `TT.py` and `EE.py` under folder `likelihoods/planck_2018_lowl`.

### Getting help offline: defaults, and bibliography

- `cobaya-citation` deprecated in favour of `cobaya-bib`. In addition to taking `.yaml` input files as below, can now take individual module names.
- `cobaya-doc` added to show defaults for particular modules.
- Added menu to `cobaya-cosmo-generator` to show defaults for modules.

### I/O

- Naming conventions for output files changed! ``*.updated.yaml`` instead of ``*.full.yaml`` for updated info, `*.[#].txt` instead of ``_[#].txt`` for chains, etc (see `Output` section in documentation).

### Samplers:

- New, more efficient minimizer: [pyBOBYQA](https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/index.html).

### Cosmology:

- Added full suite of Planck 2018 likelihoods.
- Added late-time source Cl's as a cosmological observable (CAMB only, for now)
- Changed capitalisation of some function and requests (deprecation messages and retrocompatibility added)


## 1.2.2 – 2019-08-20 (archived version)

### General

- Backported some bug fixes.
- Fixed versions of external codes.

### Cosmology:

- Planck: Fix for calibration parameter being ignored in CMBlike version of lensing likelihood.


## 1.2.0 – 2019-06-18

### General

- Added `--version` argument for `cobaya-run`
- Many bug-fixes for corner-cases

### Post-processing (still **BETA**: fixing conventions)

- Importance re-weighting, adding derived parameters, etc.

### Collections

- Now picklable!
- Support for skip and thin

### Samplers

#### Evaluate

- Multiple evaluations with new `N` option.

#### PolyChord

- Updated to version 1.16
- Handles speed-blocking optimally, including oversampling (manual blocking also possible).

### Likelihoods

- Reworked input/output parameters assignment (documented in DEVEL.rst)
- Removed deprecated `gaussian`

### Cosmo theories:

- Capitalization for observables now enforced! (fixed `H=H(z)` vs `h` ambiguity)
- CAMB and CLASS: fixed call without observables (just derived parameters)


## 1.1.3 – 2019-05-31

### Bugfixes (thanks Andreas Finke!)

### I/O

- Fuzzy-matching suggestions for options of blocks.


## 1.1.2 – 2019-05-02

### Bugfixes (thanks Vivian Miranda!)


## 1.1.1 – 2019-04-26

### I/O

- More liberal treatment of external Python objects, since we cannot check if they are the same between runs. So `force_reproducible` not needed any more! (deprecation notice left)


## 1.1.0 – 2019-04-12

### Python 3 compatibility – lots of fixes

### Cosmological likelihoods

#### Planck

- clik code updated for compatibility with Python 3 and modern gcc versions

### Cosmological theory codes

#### camb

- Updated to 1.0 (installing from master branch, considered stable)

#### classy

- Updated to ...
- Added P(k) interpolator

### Samplers

#### MCMC

- Manual parameter speed-blocking.

#### PolyChord

- Now installable with `cobaya-install polychord --modules [/path]`

### Cosmology input generator

- Added "citation" tab.


## 1.0.4 – 2019-04-11 (archived version)

### Many bugfixes -- special thanks to Guadalupe Cañas-Herrera and Vivian Miranda!

### I/O

- More permisive resuming.

### Parameterization and priors

- Made possible to fix a parameter whose only role is being an argument of a dynamically defined one.
- Likelihoods can be used in dynamical derived parameters as `chi2__[name]` (cosmological application: added automatic consolidated CMB and BAO likelihoods).

### Samplers

#### General

- Seeded runs for `evaluate`, `mcmc` and `polychord`.

#### MCMC

- Small improvements to callback functions.

#### PolyChord

- Updated to PolyChord 1.15 and using the [official GitHub repo](https://github.com/PolyChord/PolyChordLite).
- Fixed output: now -logposterior is actually that (was chi squared of posterior).
- Interfaced callback functions.

### Likelihoods

#### Created `gaussian_mixture` and added deprecation notice to `gaussian`

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
