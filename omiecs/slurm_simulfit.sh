#!/bin/bash

#SBATCH --job-name=simulfit
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=./slurm/cylsimulfit_pred.out
#SBATCH --error=./slurm/cylsimulfit_err.out
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
# module load gcc
# export MX_RCACHE=0
eval "$(conda shell.bash hook)"
conda activate micelles
echo "python from the following environment"
which python
echo "working directory = "
pwd
ulimit -s unlimited

echo "Launch Python job"
python -u ./simul_fitting.py  > ./slurm/cylsimulfit_logging.out
echo "All Done!"
exit