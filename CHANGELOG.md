## 3.0.1  – 2020-XX-XX

### Installation scripts

- File downloader function now uses `requests` instead of `wget` (less prone to segfaults), and stores intermediate files in a tmp folder.
- Added `--skip-global` option to `cobaya-install`: skips local installation of codes when the corresponding python package is available globally.
- `path=global` available for some components: forces global-scope import, even when installed with `cobaya-install`.
- Added ``--skip-not-installed`` to pytest command, to allow tests of non-installed components to fail.
- Installable components can define a class method ``is_compatible`` determining OS compatibility (assumed compatible by default). Installation of OS-incompatible components is skipped.


## 3.0  – 2020-05-12

### General

- Python 2 support removed, now requires Python 3.6+. Uses `dict` rather than `OrderedDict`.
- Significant internal refactoring including support for multiple inter-dependent theory codes.
- Greatly reduced Python overhead timing, faster for fast likelihoods.
- New base classes `CobayaComponent` and `ComponentCollection`, with support for standalone instantiation of all `CobayaComponent`.
- `.yaml` can now reference class names rather than modules, allowing multiple classes in one module.
- `.yaml` default files are now entirely at the class level, with no `kind:module:` embedding.
- inheritance of yaml and class attributes (with normal dict update, so e.g. all inherited nuisance parameters can be removed using `params:`). Each class can either define a `.yaml` or class attributes, or neither, but not both.
- The `.theory` member of likelihoods is now `Provider` class instance.
- Global `stop_at_error` option to stop at error in any component.
- Fix for more accurate timing with Python 3.
- Updates for GetDist 1.x.
- Module version information stored and checked.
- `cobaya-run --no-mpi` option to enable testing without mpi even on nodes with mpi4py installed.
- `cobaya-run-job` command to make a single job script and submit.
- docs include inheritance diagrams for key classes.
- renames `path_install` to `packages_path`, `-m` command line options to `-p`.
- `cobaya-install` saves the installation folder in a local config file. It does not need to be specified later at running, reinstalling, etc. Use `cobaya-install --show-packages-path` to show current one.
- Added `cobaya-install --skip keyword1 keyword2 ...` to skip components according to a list of keywords.
- Added citation info of Cobaya paper: [arXiv:2005.05290](https://arxiv.org/abs/2005.05290)
- Lots of other minor fixes and enhancements.

### Likelihoods and Theories

- Support for external likelihoods and theories, referenced by fully qualified package name.
- Allow referencing likelihood class names directly (`module.ClassName`).
- Ability to instantiate `Likelihood` classes directly outside Cobaya (for testing of external likelihoods or use in other packages).
- Inherited likelihoods inherit `.yaml` file from parent if no new one is defined.
- Theories and likelihoods specify requirements and define derived products with general dependencies. `get_requirements()` function replaces `add_theory()`.
- `needs()` method renamed to `must_provide()`, and can now return a dictionary of requirements conditional on those passed.
- `requires` and `provides` yaml keywords to specify which of ambiguous components handles specific requirements.
- three initialization methods: `initialize` (from `__init__`), `initialize_with_params` (after parameter assignment) and `initialize_with_provider` (once all configured).
- `Likelihood` now inherits from `Theory`, with general cached compute and `deque` states.
- `Likelihood` and `Theory` can be instantiated from `{external: class}`.
- Derived parameters in likelihood `.yaml` can be explicitly tagged with `derived:True`.
- Renamed `renames` of likelihood to `aliases` (to avoid clash with `renames` for parameters).
- Added automatic aggregated chi2 for likelihoods of the same `type`.
- More documentation for how to make internal and external likelihood classes.
- Support for `HelperTheory` classes to do sub-calculations for any `Theory` class with separate nuisance parameters and speeds.
- `classmethod` `get_class_options()` can be used to generate class defaults dynamically based on input parameters.
- Added tests: `test_dependencies.py`, `test_cosmo_multi_theory.py`.
- External likelihood functions: changed how derived parameters are specified and returned, and how externally-provided quantities are requested and obtained at run time (see docs).

### Samplers

- Samplers can now be initialized passing an already initialized model.
- Return value of `cobaya-run` now `(updated_info, sampler_instance)`. Sampler products can be retrieved as `sampler_instance.products()`.
- Sampler method now sets cache size.
- Automatic timing of likelihood and theory components to determine speed before constructing optimized blocking.
- Amount of oversampling can now be changed for MCMC and PolyChord, and it is taken into account at block sorting.
- Better dealing with files created during sampling: now all are identified and removed when `--force` used (using regexps).
- Added `cobaya-run --test` option that just initializes model and sampler.

#### MCMC

- Added progress tracking (incl. acceptance rate), and a plotting tool for it.
- Dragging now exploits blocks within slow and fast groups.

#### PolyChord

- Updated to PolyChord 1.17.1.
- Changed naming convention for raw output files, and added `getdist`-compatible `.paramnames`.
- Many defaults changes and useful documentation (Thanks Will Handley @williamjameshandley).

#### Minimize

- Support for auto-covmat as for mcmc.
- Fix for different starting points starting from existing chains using mpi.
- Fixes for bounds and rounding errors.
- Steps set from diagonal of inverse of covariance (still no use of correlation structure).
- Warnings for differences between mpi starting points.

### Cosmology

- Added `matter_power_spectrum` theory output for `z,k,P(k)` unsplined arrays.
- Fixed several bugs with `Pk_interpolator` (e.g. conflicts between likelihoods).
- `Pk_interpolator` calling arguments now different.
- Added `sigma_R` for linear rms fluctuation in sphere of radius `R`.
- Fixed problems with getting same background array theory results from different likelihoods.
- renamed `H` (array of `H(z)`) to `Hubble`.
- Boltzmann codes now consistent with varying `T_CMB`.
- changed `use_planck_names` to more general `use_renames` etc.
- DES likelihood now use numba if installed to give nearly twice faster performance.
- GUI input file generator allows to inspect auto-selected covariance matrices.

#### CAMB

- Calculation using transfer functions for speed up when only initial power spectrum and non-linear model parameters changed (even for non-linear lensing).
- Optimizations for which quantities computed.
- Option to request `CAMBdata` object from CAMB to access computed results directly.
- Fix for getting source windows power spectra.
- `external_primordial_pk` flag to optionally use a separate Cobaya Theory to return to the (binned) primordial power spectrum to CAMB.
- exposes all possible input/output parameters by introspection, making it easier to combine with other Theory classes using same parameter names.

#### CLASSY

- Updated to 2.9.3.
- Many small fixes.


## 2.0.3 – 2019-09-09

### Samplers

#### PolyChord

- Fixed too much oversampling when manual blocking (#35). Thanks Lukas Hergt (@lukashergt), Vivian Miranda (@vivianmiranda) and Will Handley (@williamjameshandley)
- Fixed ifort compatibility (#39, PR #42). Thanks Lukas Hergt (@lukashergt)

#### MCMC

- Fixed: using deprecated Pandas DataFrame method (#40). Thanks Zack Li (@xzackli)

#### Minimize

- Added GetDist output for best-fit (`ignore-prior: True`)

### Likelihoods

- Added `stop_at_error` for likelihoods -- fixes #43. Thanks Lukas Hergt (@lukashergt)

### Cosmology

- Fixed high-DPI screens (#41).


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

- More permissive resuming.

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
