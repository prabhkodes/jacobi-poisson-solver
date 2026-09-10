# FROM --platform=linux/amd64 ubuntu:22.04

# # 1. Install GCC 12, OpenMPI, UCX, and Verbs
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     gcc-12 g++-12 gcc-12-offload-nvptx \
#     openmpi-bin libopenmpi-dev \
#     libucx-dev libpmix-dev libibverbs-dev \
#     ca-certificates flex m4 libnuma-dev \
#     && rm -rf /var/lib/apt/lists/*

# # 2. Set GCC 12 as default
# RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12

# # 3. Prepare Application
# WORKDIR /app
# COPY . .

# RUN mpic++ -O3 -std=c++20 -fopenacc -foffload=-march=sm_80 \
#     -fcf-protection=none \
#     -Iinclude \
#     src/main.cpp -o app.x


    

# # 1. Use the NVIDIA HPC SDK base image
# FROM nvcr.io/nvidia/nvhpc:24.5-devel-cuda12.4-ubuntu22.04

# # 2. Set environment to use nvc++ as the MPI wrapper compiler
# ENV OMPI_CXX=nvc++
# ENV CC=nvc++
# ENV CXX=nvc++

# # 3. Setup HPC-X MPI paths (This fixes the 'opal_init' and 'help file' errors)
# # We use the explicit path found in the 24.5 image
# ENV HPCX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/comm_libs/hpcx/hpcx-2.19/ompi
# ENV OPAL_PREFIX=${HPCX_PATH}
# ENV PATH=${HPCX_PATH}/bin:${PATH}
# ENV LD_LIBRARY_PATH=${HPCX_PATH}/lib:${LD_LIBRARY_PATH}

# WORKDIR /app
# COPY . .

# # 4. Compile using the same flags as your Slurm script
# # Ubuntu 22.04's default gcc is 11.4, but nvhpc images often include 12. 
# # If it fails, remove the --gcc-toolchain flag as the base image is already compatible.
# RUN mpic++ -O3 -acc -gpu=cc80,cuda12.4 \
#     -std=c++20 \
#     -Iinclude -D_OPENACC \
#     src/main.cpp -o app.x \
#     -Minfo=acc



# 1. Use the NVIDIA HPC SDK base image
FROM nvcr.io/nvidia/nvhpc:24.5-devel-cuda12.4-ubuntu22.04

# 2. Install GCC 12 to match your Slurm module (Ubuntu 22.04 default is 11)
RUN apt-get update && apt-get install -y gcc-12 g++-12 gfortran-12 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Set environment to use nvc++ as the MPI wrapper compiler
ENV OMPI_CXX=nvc++
ENV CC=nvc++
ENV CXX=nvc++

# 4. Capture GCC 12 root (mimicking your Slurm logic)
# On Ubuntu, this is usually /usr, but we set it explicitly for the compiler
ENV GCC_ROOT=/usr

# 5. Setup HPC-X MPI paths (Fixed path for version 24.5)
ENV HPCX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/comm_libs/hpcx/hpcx-2.19/ompi
ENV OPAL_PREFIX=${HPCX_PATH}
ENV PATH=${HPCX_PATH}/bin:${PATH}
ENV LD_LIBRARY_PATH=${HPCX_PATH}/lib:${LD_LIBRARY_PATH}

WORKDIR /app
COPY . .

# 6. Compile — mirrors your Slurm flags exactly
# Added ,cuda12.4 to the -gpu flag to prevent the 'driver version 0' error
RUN mpic++ -O3 -acc -gpu=cc80,cuda12.4 \
    -std=c++20 \
    --gcc-toolchain=${GCC_ROOT} \
    -Iinclude -D_OPENACC \
    src/main.cpp -o app.x \
    -Minfo=acc

