# For release

## Thinning of output when oversampling
## version attribute should be in all components not just theory (samplers can have versions) [done for samplers; missing: likelihoods]
## Update example notebook to match example in paper
## Finish implementation of aggregated-by-data-type chi2
## No output to files while burn in makes it hard to see if working OK (default no burn?)
## turn dragging off if only one block or no speeds differ by more than factor 2
## "Not enough points in chain to check convergence" should be warning and just continue until enough
## Remove last references to odict/OrderedDict in code and *documentation*
## post: update force/resume

# Incomplete implementations/bigger jobs
## Grids/rest of cosmomc scripts
## Lots of batchjob stuff (hasConvergeBetterThan,wantCheckpointContinue etc) now broken
## containers

# cosmetic/consistency/speed

## restrict external likelihood functions to those with no requirements/theory? (new class is almost as short, and avoids syntactic inconsistencies in _theory); or use _requirements, _provider?
## use MPI for post
## In the docs "Bases" (and UML diagram) not hyperlinked correctly (not sure how to fix)
## Make numba a requirement?
## dump log info along with each chain file if saving to file (currently in stdout)
## Faster Collections for MCMC: numpy cache for merging OnePoint into Collection, `_out_update` method would take care of flushing into the Pandas table.
## PolyChord: check overhead
## PolyChord: lower dimension tests?

# Enhancements/Refactorings

## Support "parameterization" option of theory .yaml to specify parameter yaml variants?/generalize !defaults
## Let classes do all defaults combining; allow separate like instantiation + use equivalent to loading in cobaya
## check_conflicts theory method or similar (so likelihoods can raise error when used in combination with other variant likelihoods using non-independent data)
## If non-linear lensing on, model the non-linear correction via limber for faster semi-slow parameters
## unbounded parameters with flat prior (this would make it safe to rotate the unbounded ones in minimize) [JT: not very much in favour, since that would break a bunch of other stuff. Maybe let's explore an alternative solution?]
## mcmc: finish removing .checkpoint in favour of updated.yaml and .progress
## minimize: MINUIT
## minimize: maybe should not overwrite `sampler` block of original sample (either append or leave as it was)
