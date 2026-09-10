#!/bin/bash
#SBATCH --job-name="jacobi"
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition=boost_usr_prod
#SBATCH --account=ICT25_MHPC_0
#SBATCH --output=logs/jacobi_%x_%j.out 
#SBATCH --error=logs/jacobi_%x_%j.err
#SBATCH --qos=boost_qos_dbg

module purge
module load gcc/12.2.0
module load cuda
module load nvhpc
module load hpcx-mpi

# 1. FIX: Capture the path of the loaded GCC 12 module
# This ensures nvc++ finds <ranges> and other C++20 headers
GCC_ROOT=$(dirname $(dirname $(which g++)))

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 2. Compilation
# Added: --gcc-toolchain=$GCC_ROOT
nvc++ --std=c++20 --gcc-toolchain=$GCC_ROOT -O3 -o jacobi.x src/main.cpp -cuda -acc -gpu=cc80 -Minfo=acc -mp \
  -I../include \
  -L/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-torlmnyzcexnrs6pq4cccabv7ehkv3xy/Linux_x86_64/24.5/comm_libs/12.4/hpcx/hpcx-2.19/ompi/lib -lmpi \
  -L/leonardo/prod/spack/07/install/0.22/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-torlmnyzcexnrs6pq4cccabv7ehkv3xy/Linux_x86_64/24.5/cuda/lib64 -lnvToolsExt 

#4. Which modules
ldd jacobi.x

# 3. Execution
srun nsys profile \
  --trace=cuda,mpi,openacc,nvtx \
  --stats=true \
  --output=jacobi_profile_rank%q{SLURM_PROCID} \
  --force-overwrite=true \
  ./jacobi.x ./jacobian.in
