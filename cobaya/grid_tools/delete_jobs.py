import shutil
import subprocess

from . import batchjob_args, jobqueue


def delete_jobs(args=None):
    opts = batchjob_args.BatchArgs(
        "Delete running or queued jobs", importance=True, batchPathOptional=True
    )

    group = opts.parser.add_mutually_exclusive_group()
    group.add_argument("--queued", action="store_true")
    group.add_argument("--running", action="store_true")

    opts.parser.add_argument("--delete_id_min", type=int)
    opts.parser.add_argument("--delete_id_range", nargs=2, type=int)
    opts.parser.add_argument("--delete_ids", nargs="+", type=int)
    opts.parser.add_argument("--confirm", action="store_true")

    (batch, args) = opts.parseForBatch(args)

    if batch:
        if args.delete_id_range is not None:
            jobqueue.deleteJobs(
                args.batchPath, jobId_minmax=args.delete_id_range, confirm=args.confirm
            )
        if args.delete_id_min is not None:
            jobqueue.deleteJobs(
                args.batchPath, jobId_min=args.delete_id_min, confirm=args.confirm
            )
        elif args.delete_ids is not None:
            jobqueue.deleteJobs(args.batchPath, args.delete_ids, confirm=args.confirm)
        else:
            items = [jobItem for jobItem in opts.filteredBatchItems()]
            batchNames = set(
                [jobItem.name for jobItem in items]
                + [jobItem.name + "_minimize" for jobItem in items]
            )
            jobqueue.deleteJobs(
                args.batchPath, rootNames=batchNames, confirm=args.confirm
            )

        if not args.confirm:
            print("jobs not actually deleted: add --confirm to really cancel them")

    else:
        ids = []
        if args.delete_id_range is not None:
            ids = list(range(args.delete_id_range[0], args.delete_id_range[1] + 1))
        elif args.delete_ids is not None:
            ids += args.delete_ids
        elif args.name is not None:
            jobqueue.deleteJobs(args.batchPath, rootNames=args.name)
            return
        else:
            print(
                "Must give --delete_id_range, --delete_ids or --name "
                "if no batch directory"
            )
        for engine in jobqueue.grid_engine_defaults:
            qdel = jobqueue.engine_default(engine, "qdel")
            if shutil.which(qdel) is not None:
                for jobId in ids:
                    subprocess.check_output(qdel + " " + str(jobId), shell=True)
                break


if __name__ == "__main__":
    delete_jobs()
