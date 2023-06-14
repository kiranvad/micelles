#!/bin/bash

#SBATCH --job-name=micelles
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=./pred.out
#SBATCH --error=./err.out
#SBATCH --mail-user=kiranvad@uw.edu
#SBATCH --mail-type=END
#SBATCH --export=all
#SBATCH --exclusive

# usual sbatch commands
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR

eval "$(conda shell.bash hook)"
conda activate micelles
echo "python from the following environment"
which python
echo "working directory = "
pwd
ulimit -s unlimited

echo "Launch Python job"
python3 -u ./fitting.py  sph > ./logging_sph.out
python3 -u ./fitting.py  cyl > ./logging_cyl.out
python3 -u ./fitting.py  elp > ./logging_elp.out
echo "All Done!"
exit