from . import batchjob_args, jobqueue


def running_jobs(args=None):
    opts = batchjob_args.BatchArgs(
        "List details of running or queued jobs; gives job stats, "
        "then current R-1 and job/chain names",
        importance=True,
        batchPathOptional=True,
    )

    group = opts.parser.add_mutually_exclusive_group()
    group.add_argument("--queued", action="store_true")
    group.add_argument("--running", action="store_true")

    (batch, args) = opts.parseForBatch(args)

    if batch:
        items = [jobItem for jobItem in opts.filteredBatchItems()]
        batch_names = set(
            [jobItem.name for jobItem in items]
            + [jobItem.name + "_minimize" for jobItem in items]
        )
    else:
        items = None
        batch_names = set()

    ids, job_names, nameslist, infos = jobqueue.queue_job_details(
        args.batchPath, running=not args.queued, queued=not args.running
    )
    for job_id, job_name, names, info in zip(ids, job_names, nameslist, infos):
        if batch_names.intersection(names) or items is None:
            stats = dict()
            if items:
                for name in names:
                    for jobItem in items:
                        if jobItem.name == name:
                            R = jobItem.convergeStat()[0]
                            if R:
                                stats[name] = "%6.3f" % R
                            break
            R = stats.get(job_name) or " " * 6
            print(info + " |", R, job_name)
            if len(names) > 1:
                for name in names:
                    R = stats.get(name) or " " * 6
                    print("    >> ", R, name)


if __name__ == "__main__":
    running_jobs()
