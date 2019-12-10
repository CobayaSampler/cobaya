## Update example notebook to match example in paper (+ updates)
## Make numba a requirement?
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
## Check blocking/dragging when slowest block has all parameters fixed
## Option to set speeds dynamically based on actual timing for number of threads used (e.g. at first propose update)
## No output to files while burn in makes it hard to see if working OK (default no burn?)
## ,, related, how to set intermediate debugging level so see regular short output
## dump log info along with each chain file if saving to file (currently in stdout)
## check parameter default proposal widths not too large
## turn dragging off if only one block or no speeds differ by more than factor 2
## specify how to calibrate speed factors (2018 defaults looks wrong, TTTEEE slower than TT)
## If non-linear lensing on, model the non-linear correction via limber for faster semi-slow parameters
## minimize run with -f does not work. Check resuming.
## minimize with mpi and initial_point seems to start all runs at the same point??