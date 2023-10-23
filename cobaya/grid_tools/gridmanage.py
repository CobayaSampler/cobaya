import os
import fnmatch
import subprocess
import time
import shutil
import zipfile
from datetime import datetime, timedelta
import getdist
from getdist import IniFile

from cobaya.conventions import Extension
from . import batchjob_args, batchjob


# TODO: partly untested grid functions imported from cosmomc with minor changes
# might be better as cobaya-grid command [options]

def grid_check_converge(args=None):
    Opts = batchjob_args.BatchArgs('Find chains which have failed or not converged.',
                                   importance=True, converge=True)

    Opts.parser.add_argument('--exist', action='store_true')
    Opts.parser.add_argument('--checkpoint', action='store_true')
    Opts.parser.add_argument('--running', action='store_true')
    Opts.parser.add_argument('--not_running', action='store_true')
    Opts.parser.add_argument('--stuck', action='store_true')

    (batch, args) = Opts.parseForBatch(args)

    notExist = []
    converge = []

    if args.running:
        args.checkpoint = True

    if args.stuck:
        for jobItem in Opts.filteredBatchItems():
            if jobItem.chainExists() and jobItem.chainsDodgy():
                print('Chain stuck?...' + jobItem.name)
    elif args.checkpoint:
        print('Convergence from checkpoint files...')
        for jobItem in Opts.filteredBatchItems():
            R, done = jobItem.convergeStat()
            if R is not None and not done:
                if (not args.not_running or jobItem.notRunning()) and (
                        not args.running or not jobItem.notRunning()):
                    print('...', jobItem.chainRoot, R)
                if args.running and jobItem.chainExists() and jobItem.chainsDodgy():
                    print('Chain stuck?...' + jobItem.name)
    else:
        for jobItem in Opts.filteredBatchItems():
            if not jobItem.chainExists():
                notExist.append(jobItem)
            elif args.converge == 0 or args.checkpoint or not \
                    jobItem.hasConvergeBetterThan(args.converge, returnNotExist=True):
                if not args.not_running or jobItem.notRunning():
                    converge.append(jobItem)

        print('Checking batch (from last grid getdist output):')
        if not args.exist and len(notExist) > 0:
            print('Not exist...')
            for jobItem in notExist:
                print('...', jobItem.chainRoot)

        print('Converge check...')
        for jobItem in converge:
            print('...', jobItem.chainRoot, jobItem.R())


def grid_getdist(args=None):
    if isinstance(args, str):
        args = [args]

    Opts = batchjob_args.BatchArgs('Run getdist over the grid of models',
                                   notExist=True)
    Opts.parser.add_argument('--update_only', action='store_true')
    Opts.parser.add_argument('--make_plots', action='store_true',
                             help='run generated script plot files to make PDFs')
    Opts.parser.add_argument('--norun', action='store_true')
    Opts.parser.add_argument('--burn_removed', action='store_true',
                             help="if burn in has already been removed from chains")
    Opts.parser.add_argument('--burn_remove', type=float,
                             help="fraction of chain to remove as burn in "
                                  "(if not importance sampled or already done)")

    Opts.parser.add_argument('--no_plots', action='store_true',
                             help="just make non-plot outputs "
                                  "(faster if using old plot_data)")
    Opts.parser.add_argument('--delay', type=int,
                             help="run after delay of some number of seconds")
    Opts.parser.add_argument('--procs', type=int, default=1,
                             help="number of getdist instances to run in parallel")
    Opts.parser.add_argument('--base_ini', default=getdist.default_getdist_settings,
                             help="default getdist settings .ini file")
    Opts.parser.add_argument('--command', default='getdist', help="program to run")
    Opts.parser.add_argument('--exist', action='store_true',
                             help="Silently skip all chains that don't exist")

    (batch, args) = Opts.parseForBatch(args)

    ini_dir = batch.batchPath + 'getdist' + os.sep
    os.makedirs(ini_dir, exist_ok=True)

    if args.delay:
        time.sleep(args.delay)
    processes = set()

    for jobItem in Opts.filteredBatchItems():
        ini = IniFile()
        ini.params['file_root'] = jobItem.chainRoot
        ini.params['batch_path'] = jobItem.batchPath
        os.makedirs(jobItem.distPath, exist_ok=True)
        ini.params['out_dir'] = jobItem.distPath
        if os.path.exists(args.base_ini):
            ini.defaults.append(args.base_ini)
        else:
            raise ValueError("base_ini file not found")
        if hasattr(batch, 'getdist_options'):
            ini.params.update(batch.getdist_options)
        tag = ''
        if jobItem.isImportanceJob or args.burn_removed or jobItem.isBurnRemoved():
            ini.params['ignore_rows'] = 0
        elif args.burn_remove is not None:
            ini.params['ignore_rows'] = args.burn_remove

        if jobItem.isImportanceJob:
            ini.params['compare_num'] = 1
            ini.params['compare1'] = jobItem.parent.chainRoot
        if args.no_plots:
            ini.params['no_plots'] = True
        if args.make_plots:
            ini.params['make_plots'] = True
        fname = ini_dir + jobItem.name + tag + '.ini'
        ini.params.update(jobItem.dist_settings)
        ini.saveFile(fname)
        if not args.norun and (not args.notexist or not jobItem.getDistExists()) and (
                not args.update_only or jobItem.getDistNeedsUpdate()):
            if jobItem.chainExists():
                print("running: " + fname)
                processes.add(
                    subprocess.Popen([args.command] + [fname]))
                while len(processes) >= args.procs:
                    time.sleep(.1)
                    processes.difference_update(
                        [p for p in processes if p.poll() is not None])
            else:
                if not args.exist:
                    print("Chains do not exist yet: " + jobItem.chainRoot)


def grid_list(args=None):
    Opts = batchjob_args.BatchArgs('List items in a grid', importance=True, converge=True,
                                   notExist=True)
    Opts.parser.add_argument('--exists', action='store_true', help='chain must exist')
    Opts.parser.add_argument('--normed', action='store_true', help='Output normed names')

    if isinstance(args, str):
        args = [args]
    (batch, args) = Opts.parseForBatch(args)
    items = Opts.sortedParamtagDict(chainExist=args.exists)

    for paramtag, parambatch in items:
        for jobItem in parambatch:
            if hasattr(jobItem, 'group'):
                tag = '(%s)' % jobItem.group
            else:
                tag = ''
            if args.normed:
                print(jobItem.normed_name, tag)
            else:
                print(jobItem.name, tag)


def grid_cleanup(args=None):
    if isinstance(args, str):
        args = [args]

    Opts = batchjob_args.BatchArgs('delete failed chains, files etc.', importance=True,
                                   converge=True)

    Opts.parser.add_argument('--dist', action='store_true',
                             help="set to only affect getdist output files")
    Opts.parser.add_argument('--ext', nargs='+', default=['*'],
                             help="file extensions to delete")
    Opts.parser.add_argument('--empty', action='store_true')
    Opts.parser.add_argument('--confirm', action='store_true')
    Opts.parser.add_argument('--chainnum', default=None)

    (batch, args) = Opts.parseForBatch(args)

    sizeMB = 0

    def fsizestr(_fname):
        nonlocal sizeMB
        sz = os.path.getsize(_fname) // 1024
        sizeMB += sz / 1024.
        if sz < 1024:
            return str(sz) + 'KB'
        if sz < 1024 * 1024:
            return str(sz // 1024) + 'MB'
        if sz < 1024 * 1024 * 1024:
            return str(sz // 1024 // 1024) + 'GB'

    if args.chainnum is not None:
        args.ext = ['.' + args.chainnum + '.' + ext for ext in args.ext]
    else:
        args.ext = ['.' + ext for ext in args.ext] + ['.*.' + ext for ext in args.ext]

    for jobItem in Opts.filteredBatchItems():
        if (args.converge == 0 or not jobItem.hasConvergeBetterThan(
                args.converge, returnNotExist=True)) \
                and os.path.exists(jobItem.chainPath):
            dirs = [jobItem.chainPath]
            if args.dist:
                dirs = []
            if os.path.exists(jobItem.distPath):
                dirs += [jobItem.distPath]
            for adir in dirs:
                files = sorted(os.listdir(adir))
                for f in files:
                    for ext in args.ext:
                        if fnmatch.fnmatch(f, jobItem.name + ext):
                            fname = adir + f
                            if os.path.exists(fname):
                                if not args.empty or os.path.getsize(fname) == 0:
                                    print(fname, ' (' + fsizestr(fname) + ')')
                                    if args.confirm:
                                        os.remove(fname)

    print('Total size: %u MB' % int(sizeMB))
    if not args.confirm:
        print('Files not actually deleted: add --confirm to delete')


def grid_copy(args=None):

    Opts = batchjob_args.BatchArgs('copy or zip chains and optionally other files',
                                   importance=True, converge=True)

    Opts.parser.add_argument('target_dir', help="output root directory or zip file name")

    Opts.parser.add_argument('--dist', action='store_true',
                             help="include getdist outputs")
    Opts.parser.add_argument('--chains', action='store_true', help="include chain files")
    Opts.parser.add_argument('--sym_link', action='store_true',
                             help="just make symbolic links to source directories")
    Opts.parser.add_argument('--no_config', action='store_true',
                             help="don't copy grid config info")

    Opts.parser.add_argument('--remove_burn_fraction', default=0.0, type=float,
                             help="fraction at start of chain to remove as burn in")

    Opts.parser.add_argument('--file_extensions', nargs='+', default=['.*'],
                             help='extensions to include')
    Opts.parser.add_argument('--skip_extensions', nargs='+',
                             default=['.locked', '.lock_err', Extension.progress,
                                      Extension.dill, Extension.checkpoint,
                                      '.corr', '.py', '.py_mcsamples', '.pysamples'])
    Opts.parser.add_argument('--max_age_days', default=0.0, type=float,
                             help="only include files with date stamp "
                                  "at most max_age_days old")
    Opts.parser.add_argument('--dryrun', action='store_true')
    Opts.parser.add_argument('--verbose', action='store_true')
    Opts.parser.add_argument('--zip', action='store_true',
                             help='make a zip file. Not needed if target_dir '
                                  'is a filename ending in .zip')

    (batch, args) = Opts.parseForBatch(args)

    if '.zip' in args.target_dir:
        args.zip = True
    if args.max_age_days:
        max_age = datetime.now() - timedelta(days=args.max_age_days)
    else:
        max_age = None

    sizeMB = 0

    if args.zip:
        zipper = zipfile.ZipFile(args.target_dir, 'w',
                                 compression=zipfile.ZIP_DEFLATED,
                                 allowZip64=True)
        target_dir = None
    else:
        zipper = None
        target_dir = os.path.abspath(args.target_dir) + os.sep
        os.makedirs(target_dir, exist_ok=True)

    if args.sym_link and (args.remove_burn_fraction or args.zip):
        raise Exception('option not compatible with --sym_link')

    def fileMatches(_f, name):
        for ext in args.file_extensions:
            if fnmatch.fnmatch(_f, name + ext):
                for ext2 in args.skip_extensions:
                    if fnmatch.fnmatch(_f, name + ext2):
                        return False
                return True
        return False

    def doCopy(source, dest, _f, hasBurn=False):
        nonlocal sizeMB
        if args.verbose:
            print(source + _f)
        frac = 1
        if not args.dryrun:
            if args.remove_burn_fraction and hasBurn:
                lines = open(source + _f).readlines()
                lines = lines[int(len(lines) * args.remove_burn_fraction):]
                frac = 1 - args.remove_burn_fraction
            else:
                lines = None
            destf = dest + _f
            if args.zip:
                if lines:
                    zipper.writestr(destf, "".join(lines))

                else:
                    zipper.write(source + _f, destf)
            else:
                if lines:
                    open(target_dir + destf, 'w').writelines(lines)
                else:
                    if args.sym_link:
                        if os.path.islink(target_dir + destf):
                            os.unlink(target_dir + destf)
                        os.symlink(os.path.realpath(source + _f), target_dir + destf)
                    else:
                        shutil.copyfile(source + _f, target_dir + destf)
        elif args.remove_burn_fraction and hasBurn:
            frac = 1 - args.remove_burn_fraction
        sizeMB += os.path.getsize(source + _f) / 1024. ** 2 * frac

    def writeIni(iniName, _props):
        if args.dryrun:
            return
        if args.zip:
            zipper.writestr(outdir + iniName, "\n".join(_props.fileLines()))
        else:
            _props.saveFile(target_dir + outdir + iniName)

    if not args.no_config:
        config_path = os.path.join(batch.batchPath, 'config' + os.sep)
        if os.path.exists(config_path):
            if not args.dryrun and not args.zip:
                s = target_dir + 'config'
                os.makedirs(s, exist_ok=True)
            for f in os.listdir(config_path):
                doCopy(config_path, 'config' + os.sep, f)

    for jobItem in Opts.filteredBatchItems():
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
            if args.chains and jobItem.chainExists() and (
                    not args.max_age_days or
                    datetime.fromtimestamp(jobItem.chainFileDate()) > max_age):
                i = 1
                while os.path.exists(jobItem.chainRoot + '.%d.txt' % i):
                    f = jobItem.name + '.%d.txt' % i
                    chainfiles += 1
                    doCopy(jobItem.chainPath, outdir, f, not jobItem.isImportanceJob)
                    i += 1
                if not jobItem.isImportanceJob and args.remove_burn_fraction:
                    props = jobItem.propertiesIni()
                    props.params['burn_removed'] = True
                    writeIni(jobItem.name + '.properties.ini', props)
                    doneProperties = True

            for f in os.listdir(jobItem.chainPath):
                if fileMatches(f, jobItem.name):
                    if doneProperties and '.properties.ini' in f:
                        continue
                    if not args.max_age_days or datetime.fromtimestamp(
                            os.path.getmtime(jobItem.chainPath + f)) > max_age:
                        infofiles += 1
                        if args.verbose:
                            print(jobItem.chainPath + f)
                        doCopy(jobItem.chainPath, outdir, f)
            if args.dist and os.path.exists(jobItem.distPath):
                outdir += 'dist' + os.sep
                if not args.zip:
                    s2 = target_dir + outdir
                    os.makedirs(s2, exist_ok=True)
                for f in os.listdir(jobItem.distPath):
                    if fileMatches(f, jobItem.name) \
                            and (not args.max_age_days or
                                 datetime.fromtimestamp(os.path.getmtime(
                                     jobItem.distPath + f)) > max_age):
                        distfiles += 1
                        doCopy(jobItem.distPath, outdir, f)
            print('... %d chain files, %d other files and %d dist files' % (
                chainfiles, infofiles, distfiles))

    if zipper:
        zipper.close()

    print('Total size: %u MB' % sizeMB)


def grid_extract(args=None):
    Opts = batchjob_args.BatchArgs('copy all files of a given type from all getdist '
                                   'output directories in the batch',
                                   importance=True, converge=True)

    Opts.parser.add_argument('target_dir')
    Opts.parser.add_argument('file_extension', nargs='+')
    Opts.parser.add_argument('--normalize_names', action='store_true',
                             help='replace actual name tags with normalized names')
    Opts.parser.add_argument('--tag_replacements', nargs='+',
                             help="XX YY XX2 YY2 replaces name XX "
                                  "with YY, XX2 with YY2 etc.")

    (batch, args) = Opts.parseForBatch(args)

    target_dir = os.path.abspath(args.target_dir) + os.sep
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if args.tag_replacements is not None:
        replacements = dict()
        for i, val in enumerate(args.tag_replacements[::2]):
            replacements[val] = args.tag_replacements[i * 2 + 1]
    else:
        replacements = None

    for ext in args.file_extension:
        if '.' not in ext:
            pattern = '.' + ext
        else:
            pattern = ext
        for jobItem in Opts.filteredBatchItems():
            if os.path.exists(jobItem.distPath) and (
                    args.converge == 0 or jobItem.hasConvergeBetterThan(args.converge)):
                for f in os.listdir(jobItem.distPath):
                    if fnmatch.fnmatch(f, jobItem.name + pattern):
                        print(jobItem.distPath + f)
                        if args.normalize_names:
                            fout = jobItem.makeNormedName(replacements)[0] + \
                                   os.path.splitext(f)[1]
                        else:
                            fout = f
                        shutil.copyfile(jobItem.distPath + f, target_dir + fout)
