from __future__ import absolute_import, print_function, division
import subprocess
import os
import numpy as np
import re
import pickle
import time
import shutil
import multiprocessing

from distutils import spawn
import six

from cobaya.conventions import _yaml_extensions
from .conventions import _script_folder, _script_ext, _log_folder, _jobid_ext


def addArguments(parser, combinedJobs=False):
    parser.add_argument('--nodes', type=int)
    parser.add_argument('--chains-per-node', type=int)
    parser.add_argument('--cores-per-node', type=int)
    parser.add_argument('--mem-per-node', type=int, help="Memory in MB per node")
    parser.add_argument('--walltime')
    if combinedJobs:
        parser.add_argument('--combine-one-job-name',
                            help=('run all one after another, under one job submission '
                                  '(good for many fast operations)'))
        parser.add_argument('--runs-per-job', type=int,
                            default=1,
                            #                            default=int(os.environ.get('COSMOMC_runsPerJob', '1')),
                            help=('submit multiple mpi runs at once from each job script '
                                  '(e.g. to get more than one run per node)'))
    parser.add_argument('--job-template',
                        help="template file for the job submission script")
    parser.add_argument('--program', help='actual program to run (default: cobaya-run)')
    parser.add_argument('--queue', help='name of queue to submit to')
    parser.add_argument('--jobclass', help='any class name of the job')
    parser.add_argument('--qsub', help='option to change qsub command to something else')
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--no_sub', action='store_true')


def checkArguments(**kwargs):
    submitJob(None, None, msg=True, **kwargs)


class jobSettings(object):
    def __init__(self, jobName, msg=False, **kwargs):
        self.jobName = jobName
        grid_engine = 'PBS'
        if spawn.find_executable("msub") is not None:
            grid_engine = 'MOAB'
        else:
            try:
                help_info = str(
                    subprocess.check_output('qstat -help', shell=True)).strip()
                if 'OGS/GE' in help_info:
                    grid_engine = 'OGS'  # Open Grid Scheduler, as on StarCluster
            except:
                pass
        self.job_template = getDefaulted('job_template', 'job_script', **kwargs)
        if not kwargs.get("job_template"):
            raise ValueError("You must provide a script template with '--job-template'.")
        try:
            with open(self.job_template, 'r') as f:
                template = f.read()
        except IOError:
            raise ValueError("Job template '%s' not found." % self.job_template)
        try:
            cores = multiprocessing.cpu_count()
            if cores > 64:
                # probably shared memory machine, e.g. Cosmos
                cores = 8
        except:
            cores = 8
        self.cores_per_node = getDefaulted(
            'cores_per_node', cores, tp=int, template=template, **kwargs)
        if cores == 4:
            perNode = 2
        elif cores % 4 == 0:
            perNode = cores // 4
        elif cores > 3 and cores % 3 == 0:
            perNode = cores // 3
        else:
            perNode = 1
        self.chains_per_node = getDefaulted(
            'chains_per_node', perNode, tp=int, template=template, **kwargs)
        self.nodes = getDefaulted(
            'nodes', max(1, 4 // perNode), tp=int, template=template, **kwargs)
        self.nchains = self.nodes * self.chains_per_node
        self.runsPerJob = getDefaulted(
            'runsPerJob', 1, tp=int, template=template, **kwargs)
        # also defaulted at input so should be set here unless called programmatically
        self.omp = self.cores_per_node / (self.chains_per_node * self.runsPerJob)
        if self.omp != np.floor(self.omp):
            raise Exception('Chains must each have equal number of cores')
        if msg:
            print('Job parameters: %i cosmomc runs of %i chains on %i nodes, '
                  'each node with %i MPI chains, each chain using %i OpenMP cores '
                  '(%i cores per node)' % (
                      self.runsPerJob, self.nchains, self.nodes, self.chains_per_node,
                      self.omp, self.cores_per_node))
        self.mem_per_node = getDefaulted(
            'mem_per_node', 63900, tp=int, template=template, **kwargs)
        self.walltime = getDefaulted('walltime', '24:00:00', template=template, **kwargs)
        self.program = getDefaulted('program', 'cobaya-run', template=template, **kwargs)
        self.queue = getDefaulted('queue', '', template=template, **kwargs)
        self.jobclass = getDefaulted('jobclass', '', template=template, **kwargs)
        self.gridEngine = getDefaulted(
            'GridEngine', grid_engine, template=template, **kwargs)
        if grid_engine == 'OGS' and os.getenv('SGE_CLUSTER_NAME', '') == 'starcluster':
            self.qsub = 'qsub -pe orte ##NUMSLOTS##'
        else:
            self.qsub = getDefaulted(
                'qsub', 'msub' if self.gridEngine == 'MOAB' else 'qsub',
                template=template, **kwargs)
        self.qdel = getDefaulted(
            'qdel', 'canceljob' if self.gridEngine == 'MOAB' else 'qdel',
            template=template, **kwargs)
        self.runCommand = extractValue(template, 'RUN')


class jobIndex(object):
    """
    Stores the mappings between job Ids, jobNames
    """

    def __init__(self):
        self.jobSettings = dict()
        self.jobNames = dict()
        self.rootNames = dict()
        self.jobSequence = []

    def addJob(self, j):
        self.jobSettings[j.jobId] = j
        self.jobNames[j.jobName] = j.jobId
        for name in j.names:
            self.rootNames[name] = j.jobId
        self.jobSequence.append(j.jobId)

    def delId(self, jobId):
        if jobId is not None:
            j = self.jobSettings.get(jobId)
            if j is not None:
                for rootname in j.names:
                    del (self.rootNames[rootname])
                del (self.jobSettings[jobId])
                del (self.jobNames[j.jobName])
                self.jobSequence = [s for s in self.jobSequence if s != jobId]


def loadJobIndex(batchPath, must_exist=False):
    if batchPath is None:
        batchPath = os.path.join(".", _script_folder)
    fileName = os.path.join(batchPath, 'jobIndex.pyobj')
    if os.path.exists(fileName):
        with open(fileName, 'rb') as inp:
            return pickle.load(inp)
    else:
        if not must_exist:
            return jobIndex()
        return None


def saveJobIndex(obj, batchPath=None):
    if batchPath is None:
        batchPath = os.path.join(".", _script_folder)
    fname = os.path.join(batchPath, 'jobIndex.pyobj')
    with open(fname + '_tmp', 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    # try to prevent corruption from error mid-write
    if os.path.exists(fname):
        os.remove(fname)
    shutil.move(fname + '_tmp', fname)


def addJobIndex(batchPath, j):
    if batchPath is None:
        batchPath = os.path.join(".", _script_folder)
    index = loadJobIndex(batchPath)
    index.addJob(j)
    saveJobIndex(index, batchPath)


def deleteJobNames(batchPath, jobNames):
    if batchPath is None:
        batchPath = os.path.join(".", _script_folder)
    index = loadJobIndex(batchPath)
    if not index:
        raise Exception('No existing job index found')
    if isinstance(jobNames, six.string_types):
        jobNames = [jobNames]
    for name in jobNames:
        jobId = index.jobNames.get(name)
        index.delId(jobId)
    saveJobIndex(index, batchPath)


def deleteRootNames(batchPath, rootNames):
    deleteJobs(batchPath, rootNames=rootNames)


def deleteJobs(batchPath, jobIds=None, rootNames=None, jobNames=None, jobId_minmax=None,
               jobId_min=None, confirm=True, running=False, queued=False):
    if batchPath is None:
        batchPath = os.path.join(".", _script_folder)
    index = loadJobIndex(batchPath)
    if not index:
        raise Exception('No existing job index found')
    if jobIds is None:
        jobIds = []
    if isinstance(jobIds, six.string_types):
        jobIds = [jobIds]
    if rootNames is not None:
        if isinstance(rootNames, six.string_types):
            rootNames = [rootNames]
        for name in rootNames:
            jobId = index.rootNames.get(name)
            if jobId not in jobIds:
                jobIds.append(jobId)
    if jobNames is not None:
        if isinstance(jobNames, six.string_types):
            jobNames = [jobNames]
        for name in jobNames:
            jobId = index.jobNames.get(name)
            if jobId not in jobIds:
                jobIds.append(jobId)
    if jobId_minmax is not None or jobId_min is not None:
        for jobIdStr, j in list(index.jobSettings.items()):
            parts = jobIdStr.split('.')
            if len(parts) == 1 or parts[0].isdigit():
                jobId = int(parts[0])
            else:
                jobId = int(parts[1])
            if (((jobId_minmax is not None and
                  (jobId_minmax[0] <= jobId <= jobId_minmax[1]) or
                  jobId_min is not None and jobId >= jobId_min) and
                 jobIdStr not in jobIds)):
                jobIds.append(jobIdStr)
    validIds = queue_job_details(batchPath, running=not queued, queued=not running)[0]
    for jobId in jobIds:
        j = index.jobSettings.get(jobId)
        if j is not None:
            if confirm:
                if jobId in validIds:
                    print('Cancelling: ', j.jobName, jobId)
                    if hasattr(j, 'qdel'):
                        qdel = j.qdel
                    else:
                        qdel = 'qdel'
                    subprocess.check_output(qdel + ' ' + str(jobId), shell=True)
                index.delId(jobId)
            elif jobId in validIds:
                print('...', j.jobName, jobId)
    if confirm:
        saveJobIndex(index, batchPath)
    return jobIds


def submitJob(jobName, inputFiles, sequential=False, msg=False, **kwargs):
    """
    Submits a job with name ``jobName`` and input file(s) ``inputFiles``.

    If ``sequential=True`` (default: False), and multiple input files given,
    gathers the corresponding runs in a single job script, so that they are run
    sequentially.

    The ``jobSettings`` are created using the non-explicit ``kwargs``.
    """
    j = jobSettings(jobName, msg, **kwargs)
    if kwargs.get('dryrun', False) or inputFiles is None:
        return
    if isinstance(inputFiles, six.string_types):
        inputFiles = [inputFiles]
    inputFiles = [
        (os.path.splitext(p)[0] if os.path.splitext(p)[1] in _yaml_extensions else p)
        for p in inputFiles]
    j.runsPerJob = (len(inputFiles), 1)[sequential]
    # adjust omp for the actual number
    # (may not be equal to input runsPerJob because of non-integer multiple)
    j.omp = j.cores_per_node // (j.chains_per_node * j.runsPerJob)
    j.path = os.getcwd()
    j.onerun = (0, 1)[len(inputFiles) == 1 or sequential]
    vals = dict()
    vals['JOBNAME'] = jobName
    vals['OMP'] = j.omp
    vals['MEM_MB'] = j.mem_per_node
    vals['WALLTIME'] = j.walltime
    vals['NUMNODES'] = j.nodes
    vals['NUMRUNS'] = j.runsPerJob
    vals['NUMMPI'] = j.nchains
    vals['CHAINSPERNODE'] = j.chains_per_node
    vals['PPN'] = j.chains_per_node * j.runsPerJob * j.omp
    vals['MPIPERNODE'] = j.chains_per_node * j.runsPerJob
    vals['NUMTASKS'] = j.nchains * j.runsPerJob
    vals['NUMSLOTS'] = vals['PPN'] * j.nodes
    vals['ROOTDIR'] = j.path
    vals['ONERUN'] = j.onerun
    vals['PROGRAM'] = j.program
    vals['QUEUE'] = j.queue
    vals['LOGDIR'] = os.path.join(os.path.abspath(kwargs.get('batchPath')), _log_folder)
    if hasattr(j, 'jobclass'):
        vals['JOBCLASS'] = j.jobclass
    j.names = [os.path.basename(param) for param in inputFiles]
    commands = []
    for param, name in zip(inputFiles, j.names):
        ini = param + '.yaml'
        if j.runCommand is not None:
            vals['INI'] = ini
            vals['INIBASE'] = name
            command = replacePlaceholders(j.runCommand, vals)
        else:
            command = ('mpirun -np %i %s %s' % (j.nchains, j.program, ini))
        commands.append(command)
    vals['COMMAND'] = "\n".join(commands)
    with open(j.job_template, 'r') as f:
        template = f.read()
        # Remove definition lines
        template = "\n".join(
            [line for line in template.split("\n") if not line.startswith("##")])
        script = replacePlaceholders(template, vals)
        scriptRoot = os.path.join(os.path.abspath(kwargs.get('batchPath')),
                                  _script_folder, os.path.splitext(jobName)[0])
        scriptName = scriptRoot + _script_ext
        open(scriptName, 'w').write(script)
        if len(inputFiles) > 1:
            open(scriptRoot + '.batch', 'w').write("\n".join(inputFiles))
        if not kwargs.get('no_sub', False):
            res = str(subprocess.check_output(
                replacePlaceholders(j.qsub, vals) + ' ' + scriptName, shell=True)).strip()
            if not res:
                print('No qsub output')
            else:
                j.inputFiles = inputFiles
                if 'Your job ' in res:
                    m = re.search('Your job (\d*) ', res)
                    res = m.group(1)
                j.jobId = res
                j.subTime = time.time()
                open(scriptRoot + _jobid_ext, 'w').write(res)
                addJobIndex(kwargs.get('batchPath'), j)


def queue_job_details(batchPath=None, running=True, queued=True, warnNotBatch=True):
    """
    Return: list of jobIds, list of jobNames, list of list names
    """
    index = loadJobIndex(batchPath)
    if not index:
        print('No existing job index found')
        return []
    if spawn.find_executable("showq") is not None:
        res = str(subprocess.check_output('showq -U $USER', shell=True)).strip()
        runningTxt = ' Running '
    else:
        # e.g. Sun Grid Engine/OGS
        res = str(subprocess.check_output('qstat -u $USER', shell=True)).strip()
        runningTxt = ' r '
    res = res.split("\n")
    names = []
    jobNames = []
    ids = []
    infos = []
    for line in res[2:]:
        if ((' ' + os.environ.get('USER') + ' ' in line and
             (queued and not re.search(runningTxt, line, re.IGNORECASE) or
              running and re.search(runningTxt, line, re.IGNORECASE)))):
            items = line.split()
            jobId = items[0]
            j = index.jobSettings.get(jobId)
            if j is None:
                jobId = items[0].split('.')
                if jobId[0].upper() == 'TOTAL':
                    continue
                if len(jobId) == 1 or jobId[0].isdigit():
                    jobId = jobId[0]
                else:
                    jobId = jobId[1]
                j = index.jobSettings.get(jobId)
            if j is None:
                if warnNotBatch:
                    print('...Job ' + jobId + ' not in this batch, skipping')
                continue
            names += [j.names]
            jobNames += [j.jobName]
            ids += [jobId]
            infos += [line]
    return ids, jobNames, names, infos


def queue_job_names(batchPath=None, running=False, queued=True):
    lists = queue_job_details(batchPath, running, queued)[2]
    names = []
    for nameset in lists:
        names += nameset
    return names


# Functions to manipulate job templates ##################################################

def replacePlaceholders(txt, vals):
    """
    Replaces placeholders ``vals`` (dict) in a template ``txt``.
    """
    txt = txt.replace('\r', '').format(**vals)
    return txt


def extractValue(txt, name):
    """
    Extracts value of variable ``name`` defined in a template ``txt``
    as ``##name: value``.
    """
    match = re.search('##' + name + ':(.*)##', txt)
    if match:
        return match.group(1).strip()
    return None


def getDefaulted(key_name, default=None, tp=str, template=None, ext_env=None, **kwargs):
    val = kwargs.get(key_name)
    if val is None and template is not None:
        val = extractValue(template, 'DEFAULT_' + key_name)
    if val is None:
        val = os.environ.get('COSMOMC_' + key_name, None)
    if val is None and ext_env:
        val = os.environ.get('ext_env', None)
    if val is None:
        val = default
    if val is None:
        return None
    return tp(val)
