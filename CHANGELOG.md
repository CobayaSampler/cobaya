## 3.X.Y – 2022-XX-YY

### General

- Deprecated `debug_file` in input, in favour of `debug: [filename]`.
- `Prior` now has method `set_reference`, to update the reference pdf's if needed (MPI-aware).

## 3.2.1 – 2022-05-17

### General

- Fixed PyPI installation error (thanks Paul Shah!).
- Cleaner logging and better advice and error messages for missing component requirements.

## 3.2 – 2022-05-13

### General

- Documented uses of `Model` class in general contexts (previously only cosmo)
- `Model` methods to compute log-probabilities and derived parameters now have an `as_dict` keyword (default `False`), for more informative return value.
- Added `--minimize` flag to `cobaya-run` for quick minimization (replaces sampler, uses previous output).
- Add `COBAYA_USE_FILE_LOCKING` environment variable to allow disabling of file locks. Warning not to use `--test` with MPI.
- Installation of external packages is now version-aware for some packages; added `--upgrade` option to `cobaya-install`, off by default to preserve possible user changes.
- Introduced `cobaya.component.ComponentNotFoundError` to handle cases in which internal or external components cannot be found.
- In Linux terminals, added `COBAYA_COLOR` environment variable to get colourful output, useful e.g. for debugging, but *not* recommended for output to a file (e.g. running in a cluster).

### PolyChord

- Updated to v1.20.1: adds `nfail` to control failed initialisation, `synchronous` to choose sync/async parallelisation, variable number of live points, and the possibility to use an internal maximiser. Merges #232 (thanks @williamjameshandley).

### Cosmological likelihoods and theory codes

- `Pk_interpolator`: added extrapolation up to `extrap_kmin` and improved robustness

#### CAMB

- Removed problematic `zrei: zre` alias (fixes #199, thanks @pcampeti)
- Added `Omega_b|cdm|nu_massive(z)` and `angular_diameter_distance_2`
- Returned values for `get_sigma_R` changed from `R, z, sigma(z, R)` to `z, R, sigma(z, R)`.
- Support setting individual Accuracy parameters, e.g. Accuracy.AccurateBB
- Calculate accurate BB when tensors are requested
- Fix for using derived parameters with post-processing
- Added `ignore_obsolete` option to be able to run with user-modified older CAMB versions.

#### CLASS

- Updated to v3.2.0
- Added `Omega_b|cdm|nu_massive(z)`, `angular_diameter_distance_2`, `sigmaR(z)`, `sigma8(z)`, `fsgima8(z)` and Weyl potential power spectrum.
- Added `ignore_obsolete` option to be able to run with user-modified older CLASS versions.
- Added direct access to some CLASS computation products, via new requisites `CLASS_[background|thermodynamics|primordial|perturbations|sources]`.
- Changed behaviour for `non_linear`: if not present in `extra_args`, uses the current default non-linear code (HMcode) instead of no non-linear code. To impose no non-linear corrections, pass `non_linear: False`.

#### BAO

- Added Boss DR16 likelihoods (#185, by @Pablo-Lemos)

#### BICEP-Keck

- Bugfix in decorrelation function #196 (by Caterina Umilta, @umilta)
- Updated to 2021 data release (2018 data) and bugfix, #204 and #209 (by Dominic Beck, @doicbek)

#### Planck

- Fixed segfault in clik when receiving NaN in the Cl's. Partially implements #231 (thanks @lukashergt and @williamjameshandley)

## 3.1.1 – 2021-07-22

- Changes for compatibility with Pandas 1.3 (which broke convergence testing amongst other things).
- Updated docs with list of external likelihood codes, and to help avoid issues with PySide install
- Minor fixes in BAO/SN likelihoods

## 3.1 – 2021-06-04

### General

- updated and added documentation for cobaya-run-job; added cobaya-running-jobs and cobaya-delete-jobs
- Allow for more general dependencies between input parameters, derived parameters and likelihood/theory/prior inputs
- run, post and get_model can now all take inputs from a dictionary, yaml text or yaml
  filename
- Support resuming of a completed run with changed convergence parameters
- run has optional arguments to set debug, force, output, etc settings
- More input and output typing  for easier static error detection; added cobaya.typing for static checking of input dictionaries using TypedDict when available
- Refactoring of cobaya.conventions to remove most string literals and rename non-private constants starting with _
- Uses GetDist 1.2.2+ which fixes sign loading the logposterior value from Cobaya
  collection
- Optimized calculation of Gaussian 1D priors
- run settings saved to ".updated.dill_pickle" pickle file in cases where callable/class
  content cannot be preserved in yaml (install "dill")
- File locks to avoid overwriting results accidentally from multiple non-MPI processes
- Commonly-used classes can now be loaded simply using "from cobaya import Likelihood, InputDict, Theory, ..." etc., or call e.g. cobaya.run(..) 
- run and post return NamedTuples (same content as before)
- Fixed handling of "type" in external likelihood functions
- bib_script and doc_script can now be called programmatically
- MPI support refactored using decorators
- requirements can now also be specified as list of name, dictionary tuples (in case name needs to be repeated)
- renamed Collection -> SampleCollection (to avoid confusion with general typing.Collection)
- allow loading of CamelCase classes from module with lowercase name. Class "file_base_name" attribute to
  optionally specify the root name for yaml and bib files. Some supplied classes renamed.
- allow input likelihoods and theories to be instances (as well as classes); [provisional]

### MCMC

- Fixed bug with "drag: True" that gave wrong results
- MPI reworked, now avoids ending and error deadlocks, and synchronizing exceptions
  (raising OtherProcessError on non-excepting processes)
- Random number generation now using numpy 1.17 generators and MPI seeds generated using SeedSequence
  (note MPI runs generally not reproducible with fixed seed due to thead timing/asynchronous mpi exchanges)
- Overhead reduced by at least 40%, thanks to caching in Collection
- Optimization of derived parameter output (for dragging, not computed at each dragging step)
- Some refactoring/simplification to pass LogPosterior instances more 
- Reported acceptance rate is now only over last half chains (for MPI), or skipping first Rminus1_single_split fraction
- When no covamt or 'prosposal' setting for a parameter, the fallback proposal width is now scaled (narrower) from the ref or prior variance

### Post-processing

- post function reworked to support MPI, thinning, and more general parameter-dependence
  operations
- On one process operating on list of samples outputs consistent list of samples rather
  than concatenating
- Output is produced incrementally, so terminated jobs still produce valid output
- No unnecessary theory recalculations
- Support for loading from CosmoMC/Getdist-format chains.
- Function in cobaya.cosmo_input.convert_cosmomc to general Cobaya-style info from
  existing CosmoMC chains (some likelihood/theory information may have to be added if you
  are recalculating things)

### Minimize

- `PyBOBYQA` updated to 1.2, and quieter by default.
- 'best_of' parameter to probe different random starting positions (replacing seek_global_minimum for non-MPI)
- 'rhobeg' parameter larger to avoid odd hangs

### Cosmology:

- Added CamSpec 2021 Planck high-l likelihoods (based on legacy maps, not NPIPE; thanks Erik Rosenberg)
- Added Riess et al H0 constraint (H0.riess2020Mb) in terms of magnitude rather than directly on H0
  (use combined with sn.pantheon with use_abs_mag: True; thanks Pablo Lemos)
- Install updated Planck clik code (3.1) 

### Tests

- Added MPI tests and markers, synchronize errors to avoid pytest hangs on mpi errors
- Added new fast but more realistic running, resuming and post tests with and without mpi
- Fixed some randomized test inputs for more reliable running
- drag: True running test
- Coverage reporting added to Travis
- More useful traceback and console log when error raised running pytest
- added COBAYA_DEBUG env variable that can be set to force debug output (e.g. set in travis for failed build rerun)

## 3.0.4 – 2021-03-10

### General

- Added current-state-related properties to Theory (`current_state`
  replacing `_current_state` attribute, and `current_derived`
  replacing `get_current_derived()` method) and LikelihoodInterface(`current_logp`
  replacing `get_current_logp`).
- Reworked and simplified error propagation for `Theory` and `Likelihood`: clearer error
  messages and more predictable traceback printing.
- `@abstract` decorator for base classes: better control of which methods of a parent
  class have been implemented/overridden (useful e.g. for Theory classes inheriting from a
  more general one but not implementing all possible quantities that the parent class
  defines).
- For components with defaults, type annotations for class attributes now automatically
  recognised as possible input options (previously only class attributes definitions).
- Shorter parameter specification now
  possible: `<param_name>: [<prior_min>,<prior_max>,<ref_loc>,<ref_scale>,<proposal_width>]`
  , assuming a uniform prior and a normal reference pdf.
- Got up to date with changes in numpy 1.20.
- bugfix: `model.add_requirements()` does not overwrite previous calls any more.

### Cosmology:

- Interfaced sigma8 for arbitrary redshift (PR #144; thanks @Pablo-Lemos)
- Standardised naming conventions of base classes (CamelCasing, no leading underscores,
  simpler names). Added workarounds and deprecation notices for some of the old names.
- Updated cosmology `Model` example in docs.
- Added A. Lewis' CMB forecast data generator in `CMBlikes` definition file.
- `Boltzmann`: added unlensed Cl's with CAMB and CLASS.
- `CMBlikes`: small improvements, fixes, and docs.
- `InstallableLikelihood` now works with no `install_options` defined (local data).
- bugfix: bad handling of CMB polarisation capitalisation in `Boltzmann`.
- bugfix: bad `if` condition when retrieving sigmaR from `camb` (thanks @gcanasherrera and
  @matmartinelli)
- bugfix: unnecessary `camb` recomputations when setting some parameters as `extra_args`;
  fixes #142 (thanks @kimmywu)

## 3.0.3 – 2021-01-16

### General

- Bugfixes when using `cobaya.sample.get_sampler()`
- More informative error tracebacks; fixes #121 (thanks @msyriac)
- Uniform priors can now be specified simply as `[<min>, <max>]`
- Likelihoods can now be renamed and used mutiple times simultaneously; fixes #126 (thanks
  @Pablo-Lemos)

### Bibliography tools

- Bibtex files can now be specified via a class attribute, making inheritance easier (used
  to remove duplication)
- Component description now separate from bibtex code; by default, the component class
  docstring is used as description.
- Descriptions can be overridden to account for component input options (e.g. the actual
  method used in the minimizer).

### Installation scripts

- Several bugs fixed: #123, #127 and others (thanks @timothydmorton, @xgarrido)

### Minimize

- MCMC checkpoints are not deleted any more (was preventing resuming); fixes #124 (thanks
  @misharash)

### Cosmological likelihoods and theory codes

#### BAO

- Added Hubble distance and fix to `bao.generic` (Thanks @Pablo-Lemos)

#### H0

- Added Riess 2020 and Freedman et al 2020
- Normalisation changed to chi2; fixes #105 (thanks @jcolinhill)

#### CAMB

- Fixed wrong sigma8 when z=0 not requested; fixes #128, #130, #132 (thanks @Pablo-Lemos
  and @msyriac)

#### CLASS

- Fixed ignoring `l_max_scalars` (thanks Florian Stadtmann)
- Fixed #106 (thanks @lukashergt)
- Adds min gcc version check for 6.4 (thanks @williamjameshandley)

#### cosmo-generator

- Fixed PySide2 problem in newer systems; fixes #114 (thanks @talabadi)
- Fixed missing `Sampler` combo box (thanks @williamjameshandley)

## 3.0.2 – 2020-10-16

### General

- Installation bug fix.

## 3.0.1 – 2020-10-15

### General

- Cobaya can (and should!) now be called as `python -m cobaya run` instead of `cobaya-run`
  , and the same for the rest of the scripts.

### Installation scripts

- File downloader function now uses `requests` instead of `wget` (less prone to segfaults)
  , and stores intermediate files in a tmp folder.
- Added `--skip-global` option to `cobaya-install`: skips local installation of codes when
  the corresponding python package is available globally.
- `path=global` available for some components: forces global-scope import, even when
  installed with `cobaya-install`.
- Added ``--skip-not-installed`` to pytest command, to allow tests of non-installed
  components to fail.
- Installable components can define a class method ``is_compatible`` determining OS
  compatibility (assumed compatible by default). Installation of OS-incompatible
  components is skipped.

### Minimize

- Results shared with all MPI processes.
- `[prefix].updated.yaml` is now `[prefix].minimize.updated.yaml` (GetDist needs to know
  the original sampler).
- Loads covmat correcly when starting from PolyChord sample.

### Collections

- Collections are picklable again.
- Slices with omitted limits, e.g. `[::2]`, now work.
- Slicing now returns a copy of the `Collection`, instead of a raw `pandas.DataFrame`.

### MCMC

- Better MPI error handling: will now fail gracefully when called inside a user's script (
  as opposed to `cobaya-run`).

## 3.0 – 2020-05-12

### General

- Python 2 support removed, now requires Python 3.6+. Uses `dict` rather
  than `OrderedDict`.
- Significant internal refactoring including support for multiple inter-dependent theory
  codes.
- Greatly reduced Python overhead timing, faster for fast likelihoods.
- New base classes `CobayaComponent` and `ComponentCollection`, with support for
  standalone instantiation of all `CobayaComponent`.
- `.yaml` can now reference class names rather than modules, allowing multiple classes in
  one module.
- `.yaml` default files are now entirely at the class level, with no `kind:module:`
  embedding.
- inheritance of yaml and class attributes (with normal dict update, so e.g. all inherited
  nuisance parameters can be removed using `params:`). Each class can either define
  a `.yaml` or class attributes, or neither, but not both.
- The `.theory` member of likelihoods is now `Provider` class instance.
- Global `stop_at_error` option to stop at error in any component.
- Fix for more accurate timing with Python 3.
- Updates for GetDist 1.x.
- Module version information stored and checked.
- `cobaya-run --no-mpi` option to enable testing without mpi even on nodes with mpi4py
  installed.
- `cobaya-run-job` command to make a single job script and submit.
- docs include inheritance diagrams for key classes.
- renames `path_install` to `packages_path`, `-m` command line options to `-p`.
- `cobaya-install` saves the installation folder in a local config file. It does not need
  to be specified later at running, reinstalling, etc.
  Use `cobaya-install --show-packages-path` to show current one.
- Added `cobaya-install --skip keyword1 keyword2 ...` to skip components according to a
  list of keywords.
- Added citation info of Cobaya
  paper: [arXiv:2005.05290](https://arxiv.org/abs/2005.05290)
- Lots of other minor fixes and enhancements.

### Likelihoods and Theories

- Support for external likelihoods and theories, referenced by fully qualified package
  name.
- Allow referencing likelihood class names directly (`module.ClassName`).
- Ability to instantiate `Likelihood` classes directly outside Cobaya (for testing of
  external likelihoods or use in other packages).
- Inherited likelihoods inherit `.yaml` file from parent if no new one is defined.
- Theories and likelihoods specify requirements and define derived products with general
  dependencies. `get_requirements()` function replaces `add_theory()`.
- `needs()` method renamed to `must_provide()`, and can now return a dictionary of
  requirements conditional on those passed.
- `requires` and `provides` yaml keywords to specify which of ambiguous components handles
  specific requirements.
- three initialization methods: `initialize` (from `__init__`), `initialize_with_params` (
  after parameter assignment) and `initialize_with_provider` (once all configured).
- `Likelihood` now inherits from `Theory`, with general cached compute and `deque` states.
- `Likelihood` and `Theory` can be instantiated from `{external: class}`.
- Derived parameters in likelihood `.yaml` can be explicitly tagged with `derived:True`.
- Renamed `renames` of likelihood to `aliases` (to avoid clash with `renames` for
  parameters).
- Added automatic aggregated chi2 for likelihoods of the same `type`.
- More documentation for how to make internal and external likelihood classes.
- Support for `HelperTheory` classes to do sub-calculations for any `Theory` class with
  separate nuisance parameters and speeds.
- `classmethod` `get_class_options()` can be used to generate class defaults dynamically
  based on input parameters.
- Added tests: `test_dependencies.py`, `test_cosmo_multi_theory.py`.
- External likelihood functions: changed how derived parameters are specified and
  returned, and how externally-provided quantities are requested and obtained at run
  time (see docs).

### Samplers

- Samplers can now be initialized passing an already initialized model.
- Return value of `cobaya-run` now `(updated_info, sampler_instance)`. Sampler products
  can be retrieved as `sampler_instance.products()`.
- Sampler method now sets cache size.
- Automatic timing of likelihood and theory components to determine speed before
  constructing optimized blocking.
- Amount of oversampling can now be changed for MCMC and PolyChord, and it is taken into
  account at block sorting.
- Better dealing with files created during sampling: now all are identified and removed
  when `--force` used (using regexps).
- Added `cobaya-run --test` option that just initializes model and sampler.

#### MCMC

- Added progress tracking (incl. acceptance rate), and a plotting tool for it.
- Dragging now exploits blocks within slow and fast groups.

#### PolyChord

- Updated to PolyChord 1.17.1.
- Changed naming convention for raw output files, and added `getdist`
  -compatible `.paramnames`.
- Many defaults changes and useful documentation (Thanks Will Handley
  @williamjameshandley).

#### Minimize

- Support for auto-covmat as for mcmc.
- Fix for different starting points starting from existing chains using mpi.
- Fixes for bounds and rounding errors.
- Steps set from diagonal of inverse of covariance (still no use of correlation structure)
  .
- Warnings for differences between mpi starting points.

### Cosmology

- Added `matter_power_spectrum` theory output for `z,k,P(k)` unsplined arrays.
- Fixed several bugs with `Pk_interpolator` (e.g. conflicts between likelihoods).
- `Pk_interpolator` calling arguments now different.
- Added `sigma_R` for linear rms fluctuation in sphere of radius `R`.
- Fixed problems with getting same background array theory results from different
  likelihoods.
- renamed `H` (array of `H(z)`) to `Hubble`.
- Boltzmann codes now consistent with varying `T_CMB`.
- changed `use_planck_names` to more general `use_renames` etc.
- DES likelihood now use numba if installed to give nearly twice faster performance.
- GUI input file generator allows to inspect auto-selected covariance matrices.

#### CAMB

- Calculation using transfer functions for speed up when only initial power spectrum and
  non-linear model parameters changed (even for non-linear lensing).
- Optimizations for which quantities computed.
- Option to request `CAMBdata` object from CAMB to access computed results directly.
- Fix for getting source windows power spectra.
- `external_primordial_pk` flag to optionally use a separate Cobaya Theory to return to
  the (binned) primordial power spectrum to CAMB.
- exposes all possible input/output parameters by introspection, making it easier to
  combine with other Theory classes using same parameter names.

#### CLASSY

- Updated to 2.9.3.
- Many small fixes.

## 2.0.3 – 2019-09-09

### Samplers

#### PolyChord

- Fixed too much oversampling when manual blocking (#35). Thanks Lukas Hergt (@lukashergt)
  , Vivian Miranda (@vivianmiranda) and Will Handley (@williamjameshandley)
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

- Added fuzzy matching for names of modules and parameters in a few places. Now error
  messages show possible misspellings.
- Modules can now be nested, e.g. `planck_2018_lowl.TT` and `planck_2018_lowl.EE`
  as `TT.py` and `EE.py` under folder `likelihoods/planck_2018_lowl`.

### Getting help offline: defaults, and bibliography

- `cobaya-citation` deprecated in favour of `cobaya-bib`. In addition to taking `.yaml`
  input files as below, can now take individual module names.
- `cobaya-doc` added to show defaults for particular modules.
- Added menu to `cobaya-cosmo-generator` to show defaults for modules.

### I/O

- Naming conventions for output files changed! ``*.updated.yaml`` instead
  of ``*.full.yaml`` for updated info, `*.[#].txt` instead of ``_[#].txt`` for chains,
  etc (see `Output` section in documentation).

### Samplers:

- New, more efficient
  minimizer: [pyBOBYQA](https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/index.html)
  .

### Cosmology:

- Added full suite of Planck 2018 likelihoods.
- Added late-time source Cl's as a cosmological observable (CAMB only, for now)
- Changed capitalisation of some function and requests (deprecation messages and
  retrocompatibility added)

## 1.2.2 – 2019-08-20 (archived version)

### General

- Backported some bug fixes.
- Fixed versions of external codes.

### Cosmology:

- Planck: Fix for calibration parameter being ignored in CMBlike version of lensing
  likelihood.

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
- Handles speed-blocking optimally, including oversampling (manual blocking also possible)
  .

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

- More liberal treatment of external Python objects, since we cannot check if they are the
  same between runs. So `force_reproducible` not needed any more! (deprecation notice
  left)

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

- Made possible to fix a parameter whose only role is being an argument of a dynamically
  defined one.
- Likelihoods can be used in dynamical derived parameters as `chi2__[name]` (cosmological
  application: added automatic consolidated CMB and BAO likelihoods).

### Samplers

#### General

- Seeded runs for `evaluate`, `mcmc` and `polychord`.

#### MCMC

- Small improvements to callback functions.

#### PolyChord

- Updated to PolyChord 1.15 and using
  the [official GitHub repo](https://github.com/PolyChord/PolyChordLite).
- Fixed output: now -logposterior is actually that (was chi squared of posterior).
- Interfaced callback functions.

### Likelihoods

#### Created `gaussian_mixture` and added deprecation notice to `gaussian`

### Cosmological theory codes

#### General

- Added P(k) interpolator as an observable (was already available for CAMB, but not
  documented)

#### classy

- Updated to 2.7.1
- Added P(k) interpolator

### Cosmological likelihoods

#### DES

- Added Y1 release likelihoods [(arXiv:1708.01530)](https://arxiv.org/abs/1708.01530)

#### BICEP-Keck

- Updated to 2015 data [(arXiv:1810.05216)](https://arxiv.org/abs/1810.05216) and renamed
  to `bicep_keck_2015`.
