"""
.. module:: cobaya.grid_tools.gridrun

:Synopsis: script for running a grid of jobs
:Author: Antony Lewis and Jesus Torrado (based on CosmoMC version of the same code)

"""

import hashlib
import os
import shlex

from cobaya.grid_tools import batchjob_args, jobqueue
from cobaya.grid_tools.gridmanage import run_and_wait
from cobaya.tools import warn_deprecation


def grid_run(args=None):
    warn_deprecation()
    Opts = batchjob_args.BatchArgs(
        prog="cobaya-grid-run",
        desc="Submit jobs to run chains or importance sample",
        notExist=True,
        notall=True,
        converge=True,
    )
    jobqueue.addArguments(Opts.parser, combinedJobs=True)
    Opts.parser.add_argument(
        "--noqueue",
        nargs="?",
        default=None,
        type=int,
        const=1,
        help="run directly, not using queue, ignoring most other "
        "arguments (e.g. for fast tests). Optional argument "
        "specifies how many to run in parallel."
        "To use mpi, include mpirun in the --program argument.",
    )
    Opts.parser.add_argument(
        "--subitems", action="store_true", help="include sub-grid items"
    )
    Opts.parser.add_argument("--not_queued", action="store_true")
    Opts.parser.add_argument(
        "--minimize", action="store_true", help="Run minimization jobs"
    )
    Opts.parser.add_argument(
        "--importance_minimize",
        action="store_true",
        help="Run minimization jobs for chains that are importance sampled",
    )
    Opts.parser.add_argument(
        "--minimize_failed",
        action="store_true",
        help="run where minimization previously failed",
    )
    Opts.parser.add_argument(
        "--checkpoint_run",
        nargs="?",
        default=None,
        const=0,
        type=float,
        help="run if stopped and not finished; if optional value "
        "given then only run chains with convergence "
        "worse than the given value",
    )
    Opts.parser.add_argument(
        "--importance_ready",
        action="store_true",
        help="where parent chain has converged and stopped",
    )
    Opts.parser.add_argument(
        "--importance_changed",
        action="store_true",
        help="run importance jobs where the parent chain has changed since last run",
    )
    Opts.parser.add_argument(
        "--parent_converge",
        type=float,
        default=0,
        help="minimum R-1 convergence for importance job parent",
    )
    Opts.parser.add_argument(
        "--parent_stopped",
        action="store_true",
        help="only run if parent chain is not still running",
    )
    (batch, args) = Opts.parseForBatch(args)

    if args.not_queued:
        assert not args.noqueue, "Cannot use --noqueue and --not_queued"
        print("Getting queued names...")
        queued = jobqueue.queue_job_names(args.batchPath)

    def notQueued(name):
        return not any(name in job for job in queued)

    variant = ""
    if args.importance_minimize:
        variant = "_minimize"
        if args.importance is None:
            args.importance = []
    if args.minimize:
        args.noimportance = True
        variant = "_minimize"
    if args.importance is None:
        if args.importance_changed or args.importance_ready:
            args.importance = []
        else:
            args.noimportance = True
    isMinimize = args.importance_minimize or args.minimize
    if args.combine_one_job_name:
        print(
            "Combining multiple (hopefully fast) into single job script: "
            + args.combine_one_job_name
        )
    yaml_files = []
    if args.noqueue:
        program = jobqueue.get_defaulted(
            "program", default=jobqueue.default_program, program=args.program
        )
    else:
        jobqueue.check_arguments(**args.__dict__)
    processes = set()

    def jobName():
        s = "-".join([os.path.splitext(os.path.basename(ini))[0] for ini in yaml_files])
        if len(yaml_files) < 2 or len(s) < 70:
            return s
        base = os.path.basename(yaml_files[0])
        if len(base) > 70:
            base = base[:70]
        return base + "__" + hashlib.md5(s.encode("utf8")).hexdigest()[:16]

    def submitJob(ini):
        if not args.dryrun:
            print("Submitting..." + ini)
        else:
            print("... " + ini)
        if args.noqueue:
            if not args.dryrun:
                run_and_wait(processes, shlex.split(program) + [ini], args.noqueue)
            return
        yaml_files.append(ini)
        if args.combine_one_job_name:
            return
        if len(yaml_files) >= args.runs_per_job:
            if args.runs_per_job > 1:
                print("--> jobName: ", jobName())
            jobqueue.submitJob(jobName(), yaml_files, **args.__dict__)
            yaml_files.clear()

    for jobItem in Opts.filteredBatchItems(wantSubItems=args.subitems):
        if (
            (
                not args.notexist
                or isMinimize
                and not jobItem.chainMinimumExists()
                or not isMinimize
                and not jobItem.chainExists()
            )
            and (not args.minimize_failed or not jobItem.chainMinimumConverged())
            and (
                isMinimize
                or args.notall is None
                or not jobItem.allChainExists(args.notall)
            )
        ) and (not isMinimize or getattr(jobItem, "want_minimize", True)):
            if (
                not args.parent_converge
                or not jobItem.isImportanceJob
                or jobItem.parent.hasConvergeBetterThan(args.parent_converge)
            ):
                if args.converge == 0 or not jobItem.hasConvergeBetterThan(
                    args.converge, returnNotExist=True
                ):
                    if (
                        args.checkpoint_run is None
                        or jobItem.wantCheckpointContinue(args.checkpoint_run)
                        and jobItem.notRunning()
                    ):
                        if (
                            not jobItem.isImportanceJob
                            or isMinimize
                            or (
                                args.importance_ready
                                and jobItem.parent.chainFinished()
                                or not args.importance_ready
                                and jobItem.parent.chainExists()
                            )
                            and (not args.importance_changed or jobItem.parentChanged())
                            and (not args.parent_stopped or jobItem.parent.notRunning())
                        ):
                            if not args.not_queued or notQueued(jobItem.name):
                                submitJob(jobItem.yaml_file(variant))

    if args.noqueue:
        if args.noqueue > 1:
            run_and_wait(processes)
    elif yaml_files:
        if args.runs_per_job > 1:
            print("--> jobName: ", jobName())
        jobqueue.submitJob(
            args.combine_one_job_name or jobName(),
            yaml_files,
            sequential=args.combine_one_job_name is not None,
            **args.__dict__,
        )
