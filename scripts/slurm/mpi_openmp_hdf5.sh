#!/bin/bash
#SBATCH -A ICT25_MHPC
#SBATCH -p dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks-per-socket=4
#SBATCH --cpus-per-task=14
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --job-name=jacobi_pll_io_Size1000
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load hdf5  # provides HDF5_INCLUDE and HDF5_LIB

RANKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-8}
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-14}

echo "[BUILD] Compiling src/main.cpp with mpic++ + HDF5 + OpenMP …"
mpic++ src/main.cpp -std=c++20 -fopenmp \
  -I ../include -I"${HDF5_INCLUDE}" \
  -L"${HDF5_LIB}" -Wl,-rpath,"${HDF5_LIB}" \
  -lhdf5 \
  -o main.x

export OMP_NUM_THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Nodes=${SLURM_NNODES:-1}  Ranks/node=${RANKS_PER_NODE}  OMP threads=${OMP_NUM_THREADS}"
echo "Slurm: cpus-per-task=${SLURM_CPUS_PER_TASK:-?}  ntasks-per-socket=${SLURM_NTASKS_PER_SOCKET:-4}"

# ----- Require jacobian.in and log it -----
INFILE=${INFILE:-./jacobian.in}
if [[ ! -f "$INFILE" ]]; then
  echo "[ERROR] Required input file '$INFILE' not found. Aborting run."
  exit 1
fi

echo "-------------------- BEGIN ${INFILE} --------------------"
nl -ba "$INFILE"
BYTES=$(wc -c < "$INFILE" | tr -d ' ')
echo "--- (${INFILE} size: ${BYTES} bytes) ---"
(command -v md5sum >/dev/null 2>&1 && md5sum "$INFILE") || \
(command -v md5 >/dev/null 2>&1 && md5 -q "$INFILE") || true
echo "--------------------- END ${INFILE} ---------------------"

# ----- Run -----
echo "[RUN] $(date '+%F %T') — Starting to run with srun…"
srun --cpu-bind=cores ./main.x "$INFILE"
