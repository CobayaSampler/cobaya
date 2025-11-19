import multiprocessing
import os
import pickle
import re
import shutil
import subprocess
import time
from typing import Any

import numpy as np
from getdist.types import TextFile

from cobaya.conventions import Extension

from .conventions import jobid_ext, script_ext, script_folder

code_prefix = "COBAYA"
default_program = "cobaya-run -r"


def set_default_program(program, env_prefix):
    global code_prefix, default_program
    code_prefix = env_prefix
    default_program = program


def addArguments(parser, combinedJobs=False):
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--chains-per-node", type=int)
    parser.add_argument("--cores-per-node", type=int)
    parser.add_argument("--mem-per-node", type=int, help="Memory in MB per node")
    parser.add_argument("--walltime")
    if combinedJobs:
        parser.add_argument(
            "--combine-one-job-name",
            help=(
                "run all one after another, under one job submission "
                "(good for many fast operations)"
            ),
        )
        parser.add_argument(
            "--runs-per-job",
            type=int,
            default=int(os.environ.get(code_prefix + "_runsPerJob", "1")),
            help=(
                "submit multiple mpi runs at once from each job script "
                "(e.g. to get more than one run per node)"
            ),
        )
    parser.add_argument(
        "--job-template", help="template file for the job submission script"
    )
    parser.add_argument(
        "--program", help="actual program to run (default: %s)" % default_program
    )
    parser.add_argument("--queue", help="name of queue to submit to")
    parser.add_argument("--jobclass", help="any class name of the job")
    parser.add_argument("--qsub", help="option to change qsub command to something else")
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="just test configuration and give summary for checking, "
        "don't produce or do anything",
    )
    parser.add_argument(
        "--no_sub",
        action="store_true",
        help="produce job script but don't actually submit it",
    )


def check_arguments(**kwargs):
    submitJob(None, None, msg=True, **kwargs)


grid_engine_defaults = {
    "PBS": {},
    "MOAB": {"qsub": "msub", "qdel": "canceljob"},
    "SLURM": {"qsub": "sbatch", "qdel": "scancel", "qstat": "squeue"},
}


def engine_default(engine, command):
    options = grid_engine_defaults.get(engine)
    if options:
        return options.get(command, command)
    return command


class JobSettings:
    names: list
    jobId: str
    path: str
    onerun: int
    inputFiles: list
    subTime: float

    def __init__(self, jobName, msg=False, **kwargs):
        self.jobName = jobName
        grid_engine = "PBS"
        for engine, options in grid_engine_defaults.items():
            if "qsub" in options:
                if shutil.which(options["qsub"]) is not None:
                    grid_engine = engine
                    break
        else:
            if shutil.which("qstat") is not None:
                try:
                    help_info = (
                        subprocess.check_output("qstat -help", shell=True)
                        .decode()
                        .strip()
                    )
                    if "OGS/GE" in help_info:
                        grid_engine = "OGS"  # Open Grid Scheduler, as on StarCluster
                except Exception:
                    pass
        self.job_template = get_defaulted("job_template", "job_script", **kwargs)
        if (
            not self.job_template
            or self.job_template == "job_script"
            and not os.path.exists(self.job_template)
        ):
            raise ValueError(
                "You must provide a script template with '--job-template', "
                "or export COBAYA_job_template in your .bashrc"
            )
        try:
            with open(self.job_template, encoding="utf-8-sig") as f:
                template = f.read()
        except OSError:
            raise ValueError("Job template '%s' not found." % self.job_template)
        try:
            cores = multiprocessing.cpu_count()
            if cores > 64:
                # probably shared memory machine, e.g. Cosmos
                cores = 8
        except Exception:
            cores = 8
        self.cores_per_node = get_defaulted(
            "cores_per_node", cores, tp=int, template=template, **kwargs
        )
        if cores == 4:
            perNode = 2
        elif cores % 4 == 0:
            perNode = cores // 4
        elif cores > 3 and cores % 3 == 0:
            perNode = cores // 3
        else:
            perNode = 1
        self.chains_per_node = get_defaulted(
            "chains_per_node", perNode, tp=int, template=template, **kwargs
        )
        self.nodes = get_defaulted(
            "nodes", max(1, 4 // perNode), tp=int, template=template, **kwargs
        )
        self.nchains = self.nodes * self.chains_per_node
        self.runsPerJob = get_defaulted(
            "runsPerJob", 1, tp=int, template=template, **kwargs
        )
        # also defaulted at input so should be set here unless called programmatically
        self.omp = self.cores_per_node / (self.chains_per_node * self.runsPerJob)
        if self.omp != np.floor(self.omp):
            raise Exception("Chains must each have equal number of cores")
        if msg:
            print(
                "Job parameters: %i runs of %i chains on %i nodes, "
                "each node with %i MPI chains, each chain using %i OpenMP cores "
                "(%i cores per node)"
                % (
                    self.runsPerJob,
                    self.nchains,
                    self.nodes,
                    self.chains_per_node,
                    self.omp,
                    self.cores_per_node,
                )
            )
        self.mem_per_node = get_defaulted(
            "mem_per_node", 63900, tp=int, template=template, **kwargs
        )
        self.walltime = get_defaulted("walltime", "24:00:00", template=template, **kwargs)
        self.program = get_defaulted(
            "program", default_program, template=template, **kwargs
        )
        self.queue = get_defaulted("queue", "", template=template, **kwargs)
        self.jobclass = get_defaulted("jobclass", "", template=template, **kwargs)
        self.gridEngine = get_defaulted(
            "GridEngine", grid_engine, template=template, **kwargs
        )
        if grid_engine == "OGS" and os.getenv("SGE_CLUSTER_NAME", "") == "starcluster":
            self.qsub = "qsub -pe orte ##NUMSLOTS##"
        else:
            self.qsub = get_defaulted(
                "qsub", engine_default(grid_engine, "qsub"), template=template, **kwargs
            )
        # Prefer machine-readable job IDs from the scheduler where supported
        qsub_lower = self.qsub.lower()
        if grid_engine == "SLURM" or "sbatch" in qsub_lower:
            if "--parsable" not in self.qsub:
                self.qsub += " --parsable"
        elif grid_engine == "OGS":
            if "-terse" not in self.qsub:
                self.qsub += " -terse"
        self.qdel = get_defaulted(
            "qdel", engine_default(grid_engine, "qdel"), template=template, **kwargs
        )
        self.runCommand = extract_value(template, "RUN")


class JobIndex:
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
                    del self.rootNames[rootname]
                del self.jobSettings[jobId]
                del self.jobNames[j.jobName]
                self.jobSequence = [s for s in self.jobSequence if s != jobId]


def loadJobIndex(batchPath, must_exist=False):
    if batchPath is None:
        batchPath = os.path.join(".", script_folder)
    fileName = os.path.join(batchPath, "job_index.pyobj")
    if os.path.exists(fileName):
        with open(fileName, "rb") as inp:
            return pickle.load(inp)
    else:
        if not must_exist:
            return JobIndex()
        return None


def saveJobIndex(obj, batchPath=None):
    if batchPath is None:
        batchPath = os.path.join(".", script_folder)
    fname = os.path.join(batchPath, "job_index.pyobj")
    with open(fname + "_tmp", "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    # try to prevent corruption from error mid-write
    if os.path.exists(fname):
        os.remove(fname)
    shutil.move(fname + "_tmp", fname)


def addJobIndex(batchPath, j):
    if batchPath is None:
        batchPath = os.path.join(".", script_folder)
    index = loadJobIndex(batchPath)
    index.addJob(j)
    saveJobIndex(index, batchPath)


def deleteJobNames(batchPath, jobNames):
    if batchPath is None:
        batchPath = os.path.join(".", script_folder)
    index = loadJobIndex(batchPath)
    if not index:
        raise Exception("No existing job index found")
    if isinstance(jobNames, str):
        jobNames = [jobNames]
    for name in jobNames:
        jobId = index.jobNames.get(name)
        index.delId(jobId)
    saveJobIndex(index, batchPath)


def deleteRootNames(batchPath, rootNames):
    deleteJobs(batchPath, rootNames=rootNames)


def deleteJobs(
    batchPath,
    jobIds=None,
    rootNames=None,
    jobNames=None,
    jobId_minmax=None,
    jobId_min=None,
    confirm=True,
    running=False,
    queued=False,
):
    if batchPath is None:
        batchPath = os.path.join(".", script_folder)
    index = loadJobIndex(batchPath)
    if not index:
        raise Exception("No existing job index found")
    if jobIds is None:
        jobIds = []
    if isinstance(jobIds, str):
        jobIds = [jobIds]
    jobIds = set(jobIds)
    if rootNames is not None:
        if isinstance(rootNames, str):
            rootNames = [rootNames]
        jobIds.update(index.rootNames.get(name) for name in rootNames)
        jobIds.discard(None)
    if jobNames is not None:
        if isinstance(jobNames, str):
            jobNames = [jobNames]
        jobIds.update(index.jobNames.get(name) for name in jobNames)
    if jobId_minmax is not None or jobId_min is not None:
        for jobIdStr, j in list(index.jobSettings.items()):
            parts = jobIdStr.split(".")
            if len(parts) == 1 or parts[0].isdigit():
                jobId = int(parts[0])
            else:
                jobId = int(parts[1])
            if (
                jobId_minmax is not None
                and (jobId_minmax[0] <= jobId <= jobId_minmax[1])
                or jobId_min is not None
                and jobId >= jobId_min
            ):
                jobIds.add(jobIdStr)
    validIds = queue_job_details(batchPath, running=not queued, queued=not running)[0]
    for jobId in sorted(jobIds):
        j = index.jobSettings.get(jobId)
        if j is not None:
            if confirm:
                if jobId in validIds:
                    print("Cancelling: ", j.jobName, jobId)
                    if hasattr(j, "qdel"):
                        qdel = j.qdel
                    else:
                        qdel = "qdel"
                    subprocess.check_output(qdel + " " + str(jobId), shell=True)
                index.delId(jobId)
            elif jobId in validIds:
                print("...", j.jobName, jobId)
    if confirm:
        saveJobIndex(index, batchPath)
    return jobIds


def parse_job_id_from_output(res: str) -> str:
    """
    Extract a scheduler-independent job ID from a submission command's output.
    Handles common formats from SLURM, LSF, SGE/OGS, MOAB, PBS/Torque, and falls
    back to the first integer found. Returns the original string if no match.
    """
    s = res.strip()
    # Handle SLURM --parsable: JOBID[;CLUSTER]
    if ";" in s:
        first = s.split(";", 1)[0].strip()
        if first.isdigit():
            return first

    patterns = [
        r"Submitted batch job (\d+)",  # SLURM
        r"Job <(\d+)>",  # LSF
        r"Your job(?:-array)? (\d+)",  # SGE/OGS
        r"job ['\"]?(\d+)['\"]? submitted",  # MOAB variants
        r"qsub: job (\d+(?:\.[^\s]+)?)",  # PBS qsub message
        r"^(\d+(?:\.[^\s]+)?)$",  # plain ID or ID.server
    ]
    for pat in patterns:
        if m := re.search(pat, s, re.IGNORECASE):
            return m.group(1)
    if m := re.search(r"(\d+)", s):
        return m.group(1)
    return s


def submitJob(job_name, input_files, sequential=False, msg=False, **kwargs):
    """
    Submits a job with name ``jobName`` and input file(s) ``input_files``.

    If ``sequential=True`` (default: False), and multiple input files given,
    gathers the corresponding runs in a single job script, so that they are run
    sequentially.

    The ``JobSettings`` are created using the non-explicit ``kwargs``.
    """
    j = JobSettings(job_name, msg, **kwargs)
    if kwargs.get("dryrun", False) or input_files is None:
        return
    if isinstance(input_files, str):
        input_files = [input_files]
    input_files = [
        (os.path.splitext(p)[0] if os.path.splitext(p)[1] in Extension.yamls else p)
        for p in input_files
    ]
    j.runsPerJob = (len(input_files), 1)[sequential]
    # adjust omp for the actual number
    # (may not be equal to input runsPerJob because of non-integer multiple)
    j.omp = j.cores_per_node // (j.chains_per_node * j.runsPerJob)
    j.path = os.getcwd()
    j.onerun = (0, 1)[len(input_files) == 1 or sequential]
    base_path = os.path.abspath(kwargs.get("batchPath", "./"))
    script_dir = os.path.join(base_path, script_folder)
    if not os.path.exists(script_dir):
        os.makedirs(script_dir)
    vals = dict()
    vals["JOBNAME"] = job_name
    vals["OMP"] = j.omp
    vals["MEM_MB"] = j.mem_per_node
    vals["WALLTIME"] = j.walltime
    vals["NUMNODES"] = j.nodes
    vals["NUMRUNS"] = j.runsPerJob
    vals["NUMMPI"] = j.nchains
    vals["CHAINSPERNODE"] = j.chains_per_node
    vals["PPN"] = j.chains_per_node * j.runsPerJob * j.omp
    vals["MPIPERNODE"] = j.chains_per_node * j.runsPerJob
    vals["NUMTASKS"] = j.nchains * j.runsPerJob
    vals["NUMSLOTS"] = vals["PPN"] * j.nodes
    vals["ROOTDIR"] = j.path
    vals["ONERUN"] = j.onerun
    vals["PROGRAM"] = j.program
    vals["QUEUE"] = j.queue
    vals["JOBSCRIPTDIR"] = script_dir
    if hasattr(j, "jobclass"):
        vals["JOBCLASS"] = j.jobclass
    j.names = [os.path.basename(param) for param in input_files]
    commands = []
    for param, name in zip(input_files, j.names):
        ini = param + ".yaml"
        if j.runCommand is not None:
            vals["INI"] = ini
            vals["INIBASE"] = name
            command = replace_placeholders(j.runCommand, vals)
        else:
            command = "mpirun -np %i %s %s" % (j.nchains, j.program, ini)
        commands.append(command)
    vals["COMMAND"] = "\n".join(commands)
    with open(j.job_template, encoding="utf-8-sig") as f:
        template = f.read()
        # Remove definition lines
        template = "\n".join(
            [line for line in template.split("\n") if not line.startswith("##")]
        )
        script = replace_placeholders(template, vals)
        scriptRoot = os.path.join(script_dir, job_name)
        scriptName = scriptRoot + script_ext
        TextFile(scriptName).write(script)
        if len(input_files) > 1:
            TextFile(scriptRoot + ".batch").write(input_files)
        if not kwargs.get("no_sub", False):
            try:
                res = (
                    subprocess.check_output(
                        replace_placeholders(j.qsub, vals) + " " + scriptName,
                        shell=True,
                        stderr=subprocess.STDOUT,
                    )
                    .decode()
                    .strip()
                )
            except subprocess.CalledProcessError as e:
                print(f"Error calling {j.qsub}: {e.output.decode().strip()}")
            else:
                if not res:
                    print("No qsub output")
                else:
                    j.inputFiles = input_files
                    job_id = parse_job_id_from_output(res)
                    j.jobId = job_id
                    j.subTime = time.time()
                    TextFile(scriptRoot + jobid_ext).write(job_id)
                    addJobIndex(kwargs.get("batchPath"), j)


def queue_job_details(batchPath=None, running=True, queued=True, warnNotBatch=True):
    """
    Return: list of jobIds, list of job_names, list of list names
    """
    index = loadJobIndex(batchPath)
    if not index:
        print("No existing job index found")
        return []
    if shutil.which("showq") is not None:
        qstat = "showq"
        running_txt = " Running "
    else:
        if shutil.which("squeue") is not None:
            qstat = "squeue"
        else:
            # e.g. Sun Grid Engine/OGS
            qstat = "qstat"
        running_txt = " r "
    res_str = subprocess.check_output("%s -u $USER" % qstat, shell=True).decode().strip()
    res = res_str.split("\n")
    names = []
    job_names = []
    ids = []
    infos = []
    for line in res[2:]:
        if " " + str(os.environ.get("USER")) + " " in line and (
            queued
            and not re.search(running_txt, line, re.IGNORECASE)
            or running
            and re.search(running_txt, line, re.IGNORECASE)
        ):
            items = line.split()
            jobId = items[0]
            j = index.jobSettings.get(jobId)
            if j is None:
                jobId_items = items[0].split(".")
                if jobId_items[0].upper() == "TOTAL":
                    continue
                if len(jobId_items) == 1 or jobId_items[0].isdigit():
                    jobId = jobId_items[0]
                else:
                    jobId = jobId_items[1]
                j = index.jobSettings.get(jobId)
            if j is None:
                if warnNotBatch:
                    print("...Job " + jobId + " not in this batch, skipping")
                continue
            names += [j.names]
            job_names += [j.jobName]
            ids += [jobId]
            infos += [line]
    return ids, job_names, names, infos


def queue_job_names(batchPath=None, running=False, queued=True):
    lists = queue_job_details(batchPath, running, queued)[2]
    names = []
    for nameset in lists:
        names += nameset
    return names


# Functions to manipulate job templates ##################################################


def replace_placeholders(txt, vals):
    """
    Replaces placeholders ``vals`` (dict) in a template ``txt``.
    """
    txt = txt.replace("\r", "").format(**vals)
    return txt


def extract_value(txt, name):
    """
    Extracts value of variable ``name`` defined in a template ``txt``
    as ``##name: value``.
    """
    match = re.search("##" + name + ":(.*)##", txt)
    if match:
        return match.group(1).strip()
    return None


def get_defaulted(
    key_name, default=None, tp: Any = str, template=None, ext_env=None, **kwargs
) -> Any:
    val = kwargs.get(key_name)
    if val is None and template is not None:
        val = extract_value(template, "DEFAULT_" + key_name)
    if val is None:
        val = os.environ.get(code_prefix + "_" + key_name, None)
    if val is None and ext_env:
        val = os.environ.get(ext_env, None)
    if val is None:
        val = default
    if val is None:
        return None
    return tp(val)
