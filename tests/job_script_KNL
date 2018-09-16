#!/bin/bash

# Sample job template for KLN Cori @ NERSC
# Python script replaces placeholders inside CURLY BRACKETS
# with values given in the python script

# --- Job and queue info -------------------------------------------------------
#SBATCH -J {JOBNAME}
#SBATCH -q {JOBCLASS}
#SBATCH -C kln
# --- Resource allocation with optimal MPI+OpenMP interaction ------------------
#SBATCH -N {NUMNODES}
export OMP_NUM_THREADS={OMP}
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
#SBATCH -t {WALLTIME}
# --- Logging ------------------------------------------------------------------
#SBATCH -o {LOGDIR}/%x.%J.out
# --- E-mail notifications (optional) ------------------------------------------
#     (add your e-mail and remove one #+space from the following 2 lines)
# #SBATCH --mail-user=your_email@here
# #SBATCH --mail-type=ALL

# Your C/Fortran compiler + MPI + Python here
module swap craype-haswell craype-mic-knl
module load cray-mpich
module load python
export PATH=$PATH:$SCRATCH/.local/cori/2.7-anaconda-4.4/bin/

cd {ROOTDIR}

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Job ID is $SLURM_JOBID
echo Hostfile $SLURM_HOSTFILE

#Run command

{COMMAND}


## Set things to be used by the python script,
## which extracts text from here with ##XX: ...##

## Command to use for each run in the batch
##RUN: srun -n {NUMMPI} -c $(( 68 / {CHAINSPERNODE} * 4 )) --cpu_bind=cores {PROGRAM} {INI}##

## SLURM defaults
##DEFAULT_qsub: sbatch -C knl##
##DEFAULT_qdel: scancel##

## KNL Cori @ NERSC defaults
##DEFAULT_jobclass: regular##
##DEFAULT_walltime: 48:00:00##
