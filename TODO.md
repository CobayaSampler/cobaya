# Incomplete implementations/bigger jobs

## containers

# cosmetic/consistency/speed

## min/max bounds enforced on derived parameters (more generally, "bounds" as well as priors)
## make portalocker/numba/dill a requirement?
## version attribute should be in all components not just theory (samplers can have versions) [done for samplers; missing: likelihoods]
## In the docs "Bases" (and UML diagram) not hyperlinked correctly (not sure how to fix)
## dump log info along with each chain file if saving to file (currently in stdout)
## PolyChord: check overhead
## PolyChord: 
## update like/theory dict types to allow for instances (remove log.warning if all OK)
## skip measure_and_set_speeds if only one component
## maybe re-dump model info with measured speeds (need to allow saving of speeds from helper theories like camb.transfers)
## Add other sample job scripts; add auto-configure database based on cluster names

# Enhancements/Refactorings

## Way to have parameters with different speeds within the same component without splitting into separate theories or sub-HelpTheories, or specify a sub-blocking for a component's parameters
## Support "parameterization" option of theory .yaml to specify parameter yaml variants?/generalize !defaults
## Let classes do all defaults combining; allow separate like instantiation + use equivalent to loading in cobaya
## `check_conflicts` theory method or similar (so likelihoods can raise error when used in combination with other variant likelihoods using non-independent data)
## If non-linear lensing on, model the non-linear correction via limber for faster semi-slow parameters
## minimize:
+ unbounded parameters with flat prior (this would make it safe to rotate the unbounded ones in minimize) [JT: not very much in favour, since that would break a bunch of other stuff. Maybe let's explore an alternative solution? e.g. auto-extend uniform priors.]
## mcmc:
* finish removing .checkpoint in favour of updated.yaml and .progress
* For learning checks, X should perhaps ideally also depend slightly on the speed of the cycles, e.g. if either check becomes slow compared to a fast cycle.
* Update output thin factor once chains get over a given size, so that asymptotically the memory size of the chains doesn't grow indefinitely (and convergence checking time also doesn't grow correspondingly), just the thinning factor increases.
* more clever learning of covmat when only a few parameters missing: update only the row/columns of missing params, shrinkage estimator etc.
## CLASS: make it non-agnostic
## Test installed only (would need more clever pytest marking?)
## auto-covmats:
+ separate parameter matching into slow ones and fast ones, and prefer missing some fast parameters than missing slow ones.
+ refactor to more be general and don't hard code Planck etc in main source (e.g. so external likelihood distributions can provide own covmat databases)
## doc, install, model may be better documented generally rather than only in cosmo sections.
## parameterization: there should be no need for "drop" if there are no agnostic components.
## Dependencies system
* Maybe remove distinction between input parameters and requirements, so that `calculate`/`logp` takes both of them, which would be prepared by `check_cache_and_compute`. This would simplify the code a bit (in particular the part about input parameters that can be requirements, e.g. YHe) and makes all likelihood automatically callable outside a `Model` feeding requirements by hand. Problem: to prepare requirements we need arguments (e.g. units, `ell_factor` for Cl's) which are not passed to `must_compute`.
* Provider: it should be possible to save retrieving methods at initialisation so that everything (params, results, methods) can be retrieved with Provider.get(**args). Maybe it is interesting?
