#!/bin/bash

#SBATCH --job-name="jacobi"
#SBATCH --time=00:10:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition=boost_usr_prod
#SBATCH --account=ICT25_MHPC_0
#SBATCH --output=logs/jacobi_%x_%j.out 
#SBATCH --error=logs/jacobi_%x_%j.err
##SBATCH --qos=boost_qos_dbg

mkdir -p logs

module purge

# 1. Load GCC 12.2 explicitly to get C++20 support
module load gcc/12.2.0

# 2. CAPTURE THE GCC PATH
# This gets the root folder of the GCC installation (e.g., /usr/AppStream/gcc/12.2.0)
GCC_ROOT=$(dirname $(dirname $(which gcc)))
echo "[SETUP] Found GCC 12.2 root at: $GCC_ROOT"

module load cuda/12.6

module load hpcx-mpi
# 3. Load NVHPC
module load nvhpc/24.5

# 4. Load MPI
#module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2 

#5.LOAD CUDA


# --- CONFIGURATION ---
export OMPI_CXX=nvc++
export MPICH_CXX=nvc++

echo "[DEBUG] Compiler Version:"
mpic++ --version | head -n 2

echo
echo "[BUILD] Compiling..."

mpic++ -O3 -acc -gpu=cc80 \
    -std=c++20 --gcc-toolchain=$GCC_ROOT \
    -Iinclude -D_OPENACC \
    src/main.cpp -o app.x \
    -Minfo=acc 

# Error checking
if [ $? -ne 0 ]; then
    echo "===================================="
    echo "[ERROR] Compilation failed!"
    echo "===================================="
    exit 1
fi

echo "[BUILD] done."

PROCS=${SLURM_NTASKS}

echo "Nodes = $SLURM_JOB_NUM_NODES, Total Procs = $PROCS"
echo "===================================="
echo " Starting Run at $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "===================================="

INFILE="./jacobian.in"
srun ./app.x "$INFILE"

echo "===================================="
echo " End Time : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "===================================="