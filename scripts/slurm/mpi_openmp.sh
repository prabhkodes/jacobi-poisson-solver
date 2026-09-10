#!/bin/bash
#SBATCH -A ICT25_MHPC
#SBATCH -p dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8           # MPI ranks per node
#SBATCH --ntasks-per-socket=4         # split ranks across 2 sockets
#SBATCH --cpus-per-task=14            # threads per rank

##SBATCH --distribution=block:block  # pack tasks tightly per node/socket

#SBATCH --time=00:10:00
#SBATCH --job-name=jacboi_solver_Size30000
#SBATCH --output=logs/%x_%j.out

#SBATCH --hint=nomultithread          # prefer physical cores

set -euo pipefail

module purge
module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

MODE=hybrid
RANKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-8}
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-14}

which mpic++ && mpic++ --version | head -n1

echo "[BUILD] Compiling (-O0 -g, OpenMP, no OpenBLAS)…"
mpic++ -std=c++20 -O0 -g -I./include -fopenmp src/main.cpp -o app_mpi_omp.x

export OMP_NUM_THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Nodes=${SLURM_NNODES:-1}  Ranks/node=${RANKS_PER_NODE}  OMP threads=${OMP_NUM_THREADS}"
echo "Slurm: cpus-per-task=${SLURM_CPUS_PER_TASK:-?}  ntasks-per-socket=${SLURM_NTASKS_PER_SOCKET:-4}"

INFILE="./jacobian.in"
[[ -f "$INFILE" ]] || echo "WARNING: $INFILE not found; proceeding anyway."

if [[ "$MODE" == "pure" ]]; then
  export OMP_NUM_THREADS=1
  srun --cpu-bind=cores ./app_mpi_omp.x "$INFILE"
else
  srun --cpu-bind=cores ./app_mpi_omp.x "$INFILE"
fi
