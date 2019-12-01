# to check
## log.always_stop_exceptions for what exceptions to not stop_at_error and what not. 

## Update example notebook to match example in paper (+ updates)
## Update setting for model.overhead
## Already added get_version(): should add as version trace dump with output files. Where?
## Let classes do all defaults combining; allow separate like instantiation + use equivalent to loading in cobaya
## Move sampler/plik install into class methods
## Support "parameterization" option of theory .yaml to specify parameter yaml variants?
## Require py 3.7+? remove all six, odict, copy(list)..
## In the docs "Bases" (and UML diagram) not hyperlinked correctly (not sure how to fix)
## check_conflicts theory method or similar (so likelihoods can raise error when used in combination with other variant likelihoods using non-independent data)
## Finish implementation of aggregated-by-data-type chi2
## Faster Collections for MCMC: numpy cache for merging OnePoint into Collection
- `_out_update` method would take care of flushing into the Pandas table.
## check uniform bounds priors as array before calculating others

