#!/usr/bin/env python

from __future__ import absolute_import, print_function, division
import hashlib
import os

from cobaya.grid_tools import batchjob_args, jobqueue
from cobaya.tools import warn_deprecation

def run():
    warn_deprecation()
    Opts = batchjob_args.batchArgs('Submit jobs to run chains or importance sample',
                                   notExist=True, notall=True, converge=True)
    jobqueue.addArguments(Opts.parser, combinedJobs=True)
    Opts.parser.add_argument('--subitems', action='store_true',
                             help='include sub-grid items')
    Opts.parser.add_argument('--not_queued', action='store_true')
    Opts.parser.add_argument('--minimize', action='store_true',
                             help='Run minimization jobs')
    Opts.parser.add_argument('--importance_minimize', action='store_true',
                             help=('Run minimization jobs for chains '
                                   'that are importance sampled'))
    Opts.parser.add_argument('--minimize_failed', action='store_true',
                             help='run where minimization previously failed')
    Opts.parser.add_argument('--checkpoint_run', nargs='?', default=None, const=0,
                             type=float, help=(
            'run if stopped and not finished; if optional value '
            'given then only run chains with convergence worse than '
            'the given value'))
    Opts.parser.add_argument('--importance_ready', action='store_true',
                             help='where parent chain has converged and stopped')
    Opts.parser.add_argument('--importance_changed', action='store_true',
                             help=('run importance jobs where the parent chain has '
                                   'changed since last run'))
    Opts.parser.add_argument('--parent_converge', type=float, default=0,
                             help='minimum R-1 convergence for importance job parent')
    Opts.parser.add_argument('--parent_stopped', action='store_true',
                             help='only run if parent chain is not still running')
    (batch, args) = Opts.parseForBatch()
    if args.not_queued:
        print('Getting queued names...')
        queued = jobqueue.queue_job_names(args.batchPath)

    def notQueued(name):
        for job in queued:
            if name in job:
                # print 'Already running:', name
                return False
        return True

    variant = ''
    if args.importance_minimize:
        variant = '_minimize'
        if args.importance is None:
            args.importance = []
    if args.minimize:
        args.noimportance = True
        variant = '_minimize'
    if args.importance is None:
        if args.importance_changed or args.importance_ready:
            args.importance = []
        else:
            args.noimportance = True
    isMinimize = args.importance_minimize or args.minimize
    if args.combine_one_job_name:
        print('Combining multiple (hopefully fast) into single job script: ' +
              args.combine_one_job_name)
    global iniFiles
    iniFiles = []
    jobqueue.checkArguments(**args.__dict__)

    def jobName():
        s = "-".join([os.path.splitext(os.path.basename(ini))[0] for ini in iniFiles])
        if len(iniFiles) < 2 or len(s) < 70:
            return s
        base = os.path.basename(iniFiles[0])
        if len(base) > 70:
            base = base[:70]
        return base + '__' + hashlib.md5(s).hexdigest()[:16]

    def submitJob(ini):
        global iniFiles
        if not args.dryrun:
            print('Submitting...' + ini)
        else:
            print('... ' + ini)
        iniFiles.append(ini)
        if args.combine_one_job_name:
            return
        if len(iniFiles) >= args.runs_per_job:
            if args.runs_per_job > 1:
                print('--> jobName: ', jobName())
            jobqueue.submitJob(jobName(), iniFiles, **args.__dict__)
            iniFiles = []

    for jobItem in Opts.filteredBatchItems(wantSubItems=args.subitems):
        if ((not args.notexist or isMinimize and not jobItem.chainMinimumExists()
             or not isMinimize and not jobItem.chainExists()) and (
                    not args.minimize_failed or not jobItem.chainMinimumConverged())
            and (isMinimize or args.notall is None or not jobItem.allChainExists(args.notall))) \
                and (not isMinimize or getattr(jobItem, 'want_minimize', True)):
            if not args.parent_converge or not jobItem.isImportanceJob or jobItem.parent.hasConvergeBetterThan(
                    args.parent_converge):
                if args.converge == 0 or not jobItem.hasConvergeBetterThan(args.converge, returnNotExist=True):
                    if args.checkpoint_run is None or jobItem.wantCheckpointContinue(
                            args.checkpoint_run) and jobItem.notRunning():
                        if (not jobItem.isImportanceJob or isMinimize
                                or (args.importance_ready and jobItem.parent.chainFinished()
                                    or not args.importance_ready and jobItem.parent.chainExists())
                                and (not args.importance_changed or jobItem.parentChanged())
                                and (not args.parent_stopped or jobItem.parent.notRunning())):
                            if not args.not_queued or notQueued(jobItem.name):
                                submitJob(jobItem.iniFile(variant))

    if len(iniFiles) > 0:
        if args.runs_per_job > 1: print('--> jobName: ', jobName())
        jobqueue.submitJob(args.combine_one_job_name or jobName(), iniFiles,
                           sequential=args.combine_one_job_name is not None,
                           **args.__dict__)
