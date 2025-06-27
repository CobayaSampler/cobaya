"""
.. module:: cobaya.grid_tools.gridmanage

:Synopsis: tools for managing batches (grids) of chains (Cobaya version)
:Author: Antony Lewis (based on CosmoMC version of the same code)

"""

import fnmatch
import os
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timedelta

import getdist
from getdist import IniFile

from cobaya.conventions import Extension

from .batchjob_args import BatchArgs

# might be better as cobaya-grid command [options]


def run_and_wait(processes, commands=None, procs=1):
    if commands:
        processes.add(subprocess.Popen(commands))
    while len(processes) >= procs:
        time.sleep(0.1)
        processes.difference_update([p for p in processes if p.poll() is not None])


def grid_converge(args=None):
    opts = BatchArgs(
        "Find chains which have failed or not converged, and show "
        "Gelman-Rubin R-1 values for each run. Note need more than one "
        "chain for getdist to calculate R-1. "
        "Use checkpoint option to probe running chains, rather than "
        "getdist results.",
        "cobaya-grid-converge",
        importance=True,
        converge=True,
    )

    opts.parser.add_argument("--exist", action="store_true", help="chain must exist")
    opts.parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="use R-1 stored in checkpoint files (rather than getdist output)",
    )
    opts.parser.add_argument(
        "--running", action="store_true", help="only check running chains"
    )
    opts.parser.add_argument(
        "--not_running",
        action="store_true",
        help="only check chains that are not running",
    )
    opts.parser.add_argument(
        "--stuck",
        action="store_true",
        help="finds chains with big spread in the last update time",
    )

    (batch, args) = opts.parseForBatch(args)

    not_exist = []
    converge = []

    if args.running:
        args.checkpoint = True

    if args.stuck:
        for job_item in opts.filteredBatchItems():
            if job_item.chainExists() and job_item.chainsDodgy():
                print("Chain stuck?..." + job_item.name)
    elif args.checkpoint:
        print("Convergence from checkpoint files...")
        for job_item in opts.filteredBatchItems():
            R, done = job_item.convergeStat()
            if R is not None and not done:
                if (not args.not_running or job_item.notRunning()) and (
                    not args.running or not job_item.notRunning()
                ):
                    print("...", job_item.chainRoot, R)
                if args.running and job_item.chainExists() and job_item.chainsDodgy():
                    print("Chain stuck?..." + job_item.name)
    else:
        for job_item in opts.filteredBatchItems():
            if not job_item.chainExists():
                not_exist.append(job_item)
            elif (
                args.converge == 0
                or args.checkpoint
                or not job_item.hasConvergeBetterThan(args.converge, returnNotExist=True)
            ):
                if not args.not_running or job_item.notRunning():
                    converge.append(job_item)

        print("Checking batch (from last grid getdist output):")
        if not args.exist and len(not_exist) > 0:
            print("Not exist...")
            for job_item in not_exist:
                print("...", job_item.chainRoot)

        print("Converge check...")
        for job_item in converge:
            print("...", job_item.chainRoot, job_item.R())


def grid_getdist(args=None):
    opts = BatchArgs(
        "Run getdist over the grid of models. "
        "Use e.g. burn_remove=0.3 to remove 30% of the chain as burn in.",
        "cobaya-grid-getdist",
        notExist=True,
    )
    opts.parser.add_argument(
        "--update_only",
        action="store_true",
        help="only run if getdist on chains that have been updated since the last run",
    )
    opts.parser.add_argument(
        "--make_plots",
        action="store_true",
        help="run generated script plot files to make PDFs",
    )
    opts.parser.add_argument(
        "--norun",
        action="store_true",
        help="just make the .ini files, do not run getdist",
    )
    opts.parser.add_argument(
        "--burn_removed",
        action="store_true",
        help="if burn in has already been removed from chains",
    )
    opts.parser.add_argument(
        "--burn_remove",
        type=float,
        help="fraction of chain to remove as burn in "
        "(if not importance sampled or already done)",
    )

    opts.parser.add_argument(
        "--no_plots",
        action="store_true",
        help="just make non-plot outputs (faster if using old plot_data)",
    )
    opts.parser.add_argument(
        "--delay", type=int, help="run after delay of some number of seconds"
    )
    opts.parser.add_argument(
        "--procs",
        type=int,
        default=1,
        help="number of getdist instances to run in parallel",
    )
    opts.parser.add_argument(
        "--base_ini",
        default=getdist.default_getdist_settings,
        help="default getdist settings .ini file",
    )
    opts.parser.add_argument("--command", default="getdist", help="program to run")
    opts.parser.add_argument(
        "--exist", action="store_true", help="Silently skip all chains that don't exist"
    )

    (batch, args) = opts.parseForBatch(args)

    ini_dir = batch.batchPath + "getdist" + os.sep
    os.makedirs(ini_dir, exist_ok=True)

    if args.delay:
        time.sleep(args.delay)
    processes = set()

    for job_item in opts.filteredBatchItems():
        ini = IniFile()
        ini.params["file_root"] = job_item.chainRoot
        ini.params["batch_path"] = job_item.batchPath
        os.makedirs(job_item.distPath, exist_ok=True)
        ini.params["out_dir"] = job_item.distPath
        if os.path.exists(args.base_ini):
            ini.defaults.append(args.base_ini)
        else:
            raise ValueError("base_ini file not found")
        if hasattr(batch, "getdist_options"):
            ini.params.update(batch.getdist_options)
        tag = ""
        if job_item.isImportanceJob or args.burn_removed or job_item.isBurnRemoved():
            ini.params["ignore_rows"] = 0
        elif args.burn_remove is not None:
            ini.params["ignore_rows"] = args.burn_remove

        if job_item.isImportanceJob:
            ini.params["compare_num"] = 1
            ini.params["compare1"] = job_item.parent.chainRoot
        if args.no_plots:
            ini.params["no_plots"] = True
        if args.make_plots:
            ini.params["make_plots"] = True
        fname = ini_dir + job_item.name + tag + ".ini"
        ini.params.update(job_item.dist_settings)
        ini.saveFile(fname)
        if (
            not args.norun
            and (not args.notexist or not job_item.getDistExists())
            and (not args.update_only or job_item.getDistNeedsUpdate())
        ):
            if job_item.chainExists():
                print("running: " + fname)
                run_and_wait(processes, [args.command] + [fname], args.procs)
            else:
                if not args.exist:
                    print("Chains do not exist yet: " + job_item.chainRoot)

    if args.procs > 1:
        run_and_wait(processes)


def grid_list(args=None):
    opts = BatchArgs(
        "List items in a grid",
        "cobaya-grid-list",
        importance=True,
        converge=True,
        notExist=True,
    )
    opts.parser.add_argument("--exists", action="store_true", help="chain must exist")
    opts.parser.add_argument("--normed", action="store_true", help="Output normed names")

    (batch, args) = opts.parseForBatch(args)
    items = opts.sortedParamtagDict(chainExist=args.exists)

    for paramtag, parambatch in items:
        for jobItem in parambatch:
            if hasattr(jobItem, "group"):
                tag = "(%s)" % jobItem.group
            else:
                tag = ""
            if args.normed:
                print(jobItem.normed_name, tag)
            else:
                print(jobItem.name, tag)


def grid_cleanup(args=None):
    opts = BatchArgs(
        "Delete failed chains, files etc. Nothing is actually delete"
        "until you add --confirm, so you can check what you are doing first",
        "cobaya-grid-cleanup",
        importance=True,
        converge=True,
    )

    opts.parser.add_argument(
        "--dist", action="store_true", help="set to only affect getdist output files"
    )
    opts.parser.add_argument(
        "--ext", nargs="+", default=["*"], help="file extensions to delete"
    )
    opts.parser.add_argument("--empty", action="store_true")
    opts.parser.add_argument("--confirm", action="store_true")
    opts.parser.add_argument("--chainnum", default=None)

    (batch, args) = opts.parseForBatch(args)

    sizeMB = 0

    def fsizestr(_fname):
        nonlocal sizeMB
        sz = os.path.getsize(_fname) // 1024
        sizeMB += sz / 1024.0
        if sz < 1024:
            return str(sz) + "KB"
        if sz < 1024 * 1024:
            return str(sz // 1024) + "MB"
        if sz < 1024 * 1024 * 1024:
            return str(sz // 1024 // 1024) + "GB"

    if args.chainnum is not None:
        args.ext = ["." + args.chainnum + "." + ext for ext in args.ext]
    else:
        args.ext = ["." + ext for ext in args.ext] + [".*." + ext for ext in args.ext]

    done = set()
    for jobItem in opts.filteredBatchItems():
        if (
            args.converge == 0
            or not jobItem.hasConvergeBetterThan(args.converge, returnNotExist=True)
        ) and os.path.exists(jobItem.chainPath):
            dirs = [jobItem.chainPath]
            if args.dist:
                dirs = []
            if os.path.exists(jobItem.distPath):
                dirs += [jobItem.distPath]
            for adir in dirs:
                for f in sorted(os.listdir(adir)):
                    for ext in args.ext:
                        if fnmatch.fnmatch(f, jobItem.name + ext):
                            fname = adir + f
                            if fname not in done and os.path.exists(fname):
                                if not args.empty or os.path.getsize(fname) == 0:
                                    done.add(fname)
                                    print(fname, " (" + fsizestr(fname) + ")")
                                    if args.confirm:
                                        os.remove(fname)

    print("Total size: %.3g MB" % sizeMB)
    if not args.confirm:
        print("Files not actually deleted: add --confirm to delete")


def grid_copy(args=None):
    opts = BatchArgs(
        "copy or zip chains and optionally other files",
        "cobaya-grid-copy",
        importance=True,
        converge=True,
    )

    opts.parser.add_argument("target_dir", help="output root directory or zip file name")

    opts.parser.add_argument(
        "--dist", action="store_true", help="include getdist outputs"
    )
    opts.parser.add_argument("--chains", action="store_true", help="include chain files")
    opts.parser.add_argument(
        "--sym_link",
        action="store_true",
        help="just make symbolic links to source directories",
    )
    opts.parser.add_argument(
        "--no_config", action="store_true", help="don't copy grid config info"
    )

    opts.parser.add_argument(
        "--remove_burn_fraction",
        default=0.0,
        type=float,
        help="fraction at start of chain to remove as burn in",
    )

    opts.parser.add_argument(
        "--file_extensions", nargs="+", default=[".*"], help="extensions to include"
    )
    opts.parser.add_argument(
        "--skip_extensions",
        nargs="+",
        default=[
            ".locked",
            ".lock_err",
            Extension.progress,
            Extension.dill,
            Extension.checkpoint,
            ".corr",
            ".py",
            ".py_mcsamples",
            ".pysamples",
        ],
    )
    opts.parser.add_argument(
        "--max_age_days",
        default=0.0,
        type=float,
        help="only include files with date stamp at most max_age_days old",
    )
    opts.parser.add_argument("--dryrun", action="store_true")
    opts.parser.add_argument("--verbose", action="store_true")
    opts.parser.add_argument(
        "--zip",
        action="store_true",
        help="make a zip file. Not needed if target_dir is a filename ending in .zip",
    )

    (batch, args) = opts.parseForBatch(args)

    if args.target_dir.endswith(".zip"):
        args.zip = True
    if args.max_age_days:
        max_age = datetime.now() - timedelta(days=args.max_age_days)
    else:
        max_age = None

    sizeMB = 0

    if args.zip:
        zipper = zipfile.ZipFile(
            args.target_dir, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        )
        target_dir = None
    else:
        zipper = None
        target_dir = os.path.abspath(args.target_dir) + os.sep
        os.makedirs(target_dir, exist_ok=True)

    if args.sym_link and (args.remove_burn_fraction or args.zip):
        raise Exception("option not compatible with --sym_link")

    def file_matches(_f, name):
        for ext in args.file_extensions:
            if fnmatch.fnmatch(_f, name + ext):
                for ext2 in args.skip_extensions:
                    if fnmatch.fnmatch(_f, name + ext2):
                        return False
                return True
        return False

    done = set()

    def do_copy(source, dest, _f, has_burn=False):
        nonlocal sizeMB
        if args.verbose:
            print(source + _f)
        frac = 1
        if not args.dryrun:
            if args.remove_burn_fraction and has_burn:
                lines = open(source + _f, encoding="utf-8-sig").readlines()
                lines = lines[: int((len(lines)) * args.remove_burn_fraction) :]
                frac = 1 - args.remove_burn_fraction
            else:
                lines = None
            destf = dest + _f
            if destf in done:
                return
            done.add(destf)
            if args.zip:
                if lines:
                    zipper.writestr(destf, "".join(lines))
                else:
                    zipper.write(source + _f, destf)
            else:
                if lines:
                    open(target_dir + destf, "w", encoding="utf-8").writelines(lines)
                else:
                    if args.sym_link:
                        if os.path.islink(target_dir + destf):
                            os.unlink(target_dir + destf)
                        os.symlink(os.path.realpath(source + _f), target_dir + destf)
                    else:
                        shutil.copyfile(source + _f, target_dir + destf)
        elif args.remove_burn_fraction and has_burn:
            frac = 1 - args.remove_burn_fraction
        sizeMB += os.path.getsize(source + _f) / 1024.0**2 * frac

    def write_ini(iniName, _props):
        if args.dryrun:
            return
        if args.zip:
            zipper.writestr(outdir + iniName, "\n".join(_props.fileLines()))
        else:
            _props.saveFile(target_dir + outdir + iniName)

    if not args.no_config:
        config_path = os.path.join(batch.batchPath, "config" + os.sep)
        if os.path.exists(config_path):
            if not args.dryrun and not args.zip:
                s = target_dir + "config"
                os.makedirs(s, exist_ok=True)
            for f in os.listdir(config_path):
                do_copy(config_path, "config" + os.sep, f)

    for jobItem in opts.filteredBatchItems():
        if args.converge == 0 or jobItem.hasConvergeBetterThan(args.converge):
            print(jobItem.name)
            chainfiles = 0
            infofiles = 0
            distfiles = 0
            doneProperties = False
            outdir = jobItem.relativePath
            if not args.zip:
                s1 = target_dir + outdir
                os.makedirs(s1, exist_ok=True)
            if (
                args.chains
                and jobItem.chainExists()
                and (
                    not args.max_age_days
                    or datetime.fromtimestamp(jobItem.chainFileDate()) > max_age
                )
            ):
                i = 1
                while os.path.exists(jobItem.chainRoot + ".%d.txt" % i):
                    f = jobItem.name + ".%d.txt" % i
                    chainfiles += 1
                    do_copy(jobItem.chainPath, outdir, f, not jobItem.isImportanceJob)
                    i += 1
                if not jobItem.isImportanceJob and args.remove_burn_fraction:
                    props = jobItem.propertiesIni()
                    props.params["burn_removed"] = True
                    write_ini(jobItem.name + ".properties.ini", props)
                    doneProperties = True

            for f in os.listdir(jobItem.chainPath):
                if file_matches(f, jobItem.name):
                    if doneProperties and ".properties.ini" in f:
                        continue
                    if (
                        not args.max_age_days
                        or datetime.fromtimestamp(os.path.getmtime(jobItem.chainPath + f))
                        > max_age
                    ):
                        infofiles += 1
                        if args.verbose:
                            print(jobItem.chainPath + f)
                        do_copy(jobItem.chainPath, outdir, f)
            if args.dist and os.path.exists(jobItem.distPath):
                outdir += "dist" + os.sep
                if not args.zip:
                    s2 = target_dir + outdir
                    os.makedirs(s2, exist_ok=True)
                for f in os.listdir(jobItem.distPath):
                    if file_matches(f, jobItem.name) and (
                        not args.max_age_days
                        or datetime.fromtimestamp(os.path.getmtime(jobItem.distPath + f))
                        > max_age
                    ):
                        distfiles += 1
                        do_copy(jobItem.distPath, outdir, f)
            print(
                "... %d chain files, %d other files and %d dist files"
                % (chainfiles, infofiles, distfiles)
            )

    if zipper:
        zipper.close()

    print("Total size: %u MB" % sizeMB)


def grid_extract(args=None):
    opts = BatchArgs(
        "copy all files of a given type from all getdist output directories in the grid",
        "cobaya-grid-extract",
        importance=True,
        converge=True,
    )

    opts.parser.add_argument("target_dir")
    opts.parser.add_argument("file_extension", nargs="+")
    opts.parser.add_argument(
        "--normalize_names",
        action="store_true",
        help="replace actual name tags with normalized names",
    )
    opts.parser.add_argument(
        "--tag_replacements",
        nargs="+",
        help="XX YY XX2 YY2 replaces name XX with YY, XX2 with YY2 etc.",
    )

    (batch, args) = opts.parseForBatch(args)

    target_dir = os.path.abspath(args.target_dir) + os.sep
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if args.tag_replacements is not None:
        replacements = {}
        for i, val in enumerate(args.tag_replacements[::2]):
            replacements[val] = args.tag_replacements[i * 2 + 1]
    else:
        replacements = None

    for ext in args.file_extension:
        if "." not in ext:
            pattern = "." + ext
        else:
            pattern = ext
        for jobItem in opts.filteredBatchItems():
            if os.path.exists(jobItem.distPath) and (
                args.converge == 0 or jobItem.hasConvergeBetterThan(args.converge)
            ):
                for f in os.listdir(jobItem.distPath):
                    if fnmatch.fnmatch(f, jobItem.name + pattern):
                        print(jobItem.distPath + f)
                        if args.normalize_names:
                            file_out = (
                                jobItem.makeNormedName(replacements)[0]
                                + os.path.splitext(f)[1]
                            )
                        else:
                            file_out = f
                        shutil.copyfile(jobItem.distPath + f, target_dir + file_out)
