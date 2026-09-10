# Parallel Programming Exam

## Matrix Multiplication

Hybrid (MPI + OpenMP)

```jsx
Matrix Size = 22,400 * 22,400
Timer Precision = milliseconds

Slurm Config
#!/bin/bash
#SBATCH -A ICT25_MHPC
#SBATCH -p dcgp_usr_prod
#SBATCH --nodes=N
#SBATCH --ntasks-per-node=112
#SBATCH --time=00:10:00
#SBATCH --job-name=mat_mult_hybrid
#SBATCH --output=logs/%x_%j.out

# hybrid combinations: 56*2,28*4,14*8
RANKS_PER_NODE=2     
OMP_NUM_THREADS=56     

export OMP_NUM_THREADS=${OMP_NUM_THREADS}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

Compile Command
mpic++ -std=c++20 -O3 -march=native -Wall -Wextra -I./include -fopenmp \
       src/main.cpp -lopenblas -o app_mpi_omp.x
 
Run Command
  mpirun --bind-to core --map-by ppr:${RANKS_PER_NODE}:node:pe=${OMP_NUM_THREADS} \
         -n ${PROCS} ./app_mpi_omp.x
         

Raw Data

Label                                   Ranks_per_node  MPI_ranks_total  OMP_threads  DGEMM  MPI_AllGather  Extract_B  INIT_A_B  Write_C
"N=1 | rpn=28 | mpi=28 | omp=4"         28              28               4            3637   1262           2          268       27
"N=1 | rpn=56 | mpi=56 | omp=2"         56              56               2            3843   1694           0          144        0
"N=1 | rpn=14 | mpi=14 | omp=8"         14              14               8            3489   1152          21          510       55
"N=1 | rpn=4  | mpi=4  | omp=28"         4               4              28             293    118           5         1775        8
"N=1 | rpn=2  | mpi=2  | omp=56"         2               2              56            1402    153         105         3547       76
"N=1 | rpn=8  | mpi=8  | omp=14"         8               8              14             102     53           0          889        0

"N=2 | rpn=28 | mpi=56  | omp=4"        28              56               4            2072   1078           0          139        0
"N=2 | rpn=56 | mpi=112 | omp=2"        56             112               2            2256   1338           0           73        0
"N=2 | rpn=14 | mpi=28  | omp=8"        14              28               8            1931    933           2          256       23
"N=2 | rpn=4  | mpi=8   | omp=28"        4               8              28              68     32           0          891        0
"N=2 | rpn=2  | mpi=4   | omp=56"        2               4              56             267     60           5         1778        8
"N=2 | rpn=8  | mpi=16  | omp=14"        8              16              14            1838    944           6          446       32

"N=4 | rpn=28 | mpi=112 | omp=4"        28              112              4            1462    867           0           76        0
"N=4 | rpn=56 | mpi=224 | omp=2"        56              224              2            1638   1385           0           37        0
"N=4 | rpn=14 | mpi=56  | omp=8"        14               56              8            1254    790           0          129        0
"N=4 | rpn=4  | mpi=16  | omp=28"        4               16             28            1169    926           5          462       32
"N=4 | rpn=2  | mpi=8   | omp=56"        2                8             56              70     28           0          891        0
"N=4 | rpn=8  | mpi=32  | omp=14"        8               32             14            1147    799           1          233        0

"N=8 | rpn=14 | mpi=112 | omp=8"        14              112              8             998    762           0          67        0
"N=8 | rpn=28 | mpi=224 | omp=4"        28              224              4            1339   1065           0          39        0
"N=8 | rpn=56 | mpi=448 | omp=2"        56              448              2            1355   1411           0          21        0
"N=8 | rpn=8  | mpi=64  | omp=14"        8               64             14             821    696           0         118        0
"N=8 | rpn=4  | mpi=32  | omp=28"        4               32             28             923    820           1         233        0
"N=8 | rpn=2  | mpi=16  | omp=56"        2               16             56            1049    800           5         448       32

"N=10 | rpn=2  | mpi=20  | omp=56"        2               20              56             28     30           2          360        0
"N=10 | rpn=4  | mpi=40  | omp=28"        4               40              28              4     12           0          181        0
"N=10 | rpn=8  | mpi=80  | omp=14"        8               80              14            732    667           3          105        0
"N=10 | rpn=14 | mpi=140 | omp=8"        14              140               8            985    834           0           55        0
"N=10 | rpn=28 | mpi=280 | omp=4"        28              280               4           1130   1238           0           32        0

"N=16 | rpn=14 | mpi=224 | omp=8"       14              224              8            684    782           2           37        0
"N=16 | rpn=28 | mpi=448 | omp=4"       28              448              4            902   1195           0           21        0
"N=16 | rpn=8  | mpi=128 | omp=14"       8              128             14            650    727           0           58        0
"N=16 | rpn=4  | mpi=64  | omp=28"       4               64             28            518    765           0          115        0
"N=16 | rpn=2  | mpi=32  | omp=56"       2               32             56            581    723           3          227        0
 
```

![image.png](Parallel%20Programming%20Exam/image.png)

```jsx
Performance Benchmark on Leonardo
Nodes=16  Ranks/node=8  MPI ranks=128  OMP threads=14
Matrix N = 224,000
Starting execution at rank 0 || 2025-11-13 16:47:37.303
Ending execution at rank 0 || 2025-11-13 16:56:02.760

----------------------------------------------------------------------------------------------------
Function : DGEMM                Max time: 217077ms Min time: 214010ms Avg time : 215496ms
----------------------------------------------------------------------------------------------------
Function : Extract B            Max time:   139ms Min time:   135ms Avg time : 135.25ms
----------------------------------------------------------------------------------------------------
Function : INIT A,B             Max time:  5540ms Min time:  5515ms Avg time : 5528.07ms
----------------------------------------------------------------------------------------------------
Function : MPI AllGather        Max time: 281196ms Min time: 278030ms Avg time : 279657ms
----------------------------------------------------------------------------------------------------
Function : Write to C           Max time:   450ms Min time:   384ms Avg time : 385.508ms
----------------------------------------------------------------------------------------------------

Time(seconds) = (217077 ms + 139 ms + 281196 ms + 450 ms) / 1000
              = 498.862 seconds  
              ~= 499 seconds

Flops = 224,000 ^ 3 * 2 / time
      = 224788e16 / 499
      = 22,478e12 / 499
      = 44,956e12 
      ~= 45 TFlops
```

---

## Jacobi Solver

Hybrid (MPI + OpenMP)

```jsx
Grid Size = 10,000 * 10,000
Timer Precision = microseconds

Slurm Configuration
#!/bin/bash
#SBATCH -A ICT25_MHPC
#SBATCH -p dcgp_usr_prod
#SBATCH --nodes=10

##hybrid: 56(2t),28(4t),14(8t)
#SBATCH --ntasks-per-node=2          
#SBATCH --cpus-per-task=56           
#SBATCH --ntasks-per-socket=1        
#SBATCH --time=00:10:00
#SBATCH --job-name=jacboi_solver
#SBATCH --output=logs/%x_%j.out
#SBATCH --hint=nomultithread   

RANKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-8}
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-14}

export OMP_NUM_THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

Compile Command 
# No optimisations
mpic++ -std=c++20 -O0 -g -I./include -fopenmp src/main.cpp -o app_mpi_omp.x
 
Run Command
srun --cpu-bind=cores ./app_mpi_omp.x "$INFILE"

Raw Data
Label                               Ranks_per_node MPI_ranks_total OMP_threads     EXCHANGE_BOUNDARIES  INIT_fields_with_Halo      SOLVE_AND_SWAP 
"N=1 | rpn=2  | mpi=2  | omp=56"        2               2               56              1834309               386677                  34524307         
"N=1 | rpn=4  | mpi=4  | omp=28"        4               4               28               958298               195283                  18511096         
"N=1 | rpn=8  | mpi=8  | omp=14"        8               8               14               843351                99449                  17027085         
"N=1 | rpn=14 | mpi=14 | omp=8"        14              14                8              1068969                57677                  17352688         
"N=1 | rpn=28 | mpi=28 | omp=4"        28              28                4               792254                31619                  16650145         
"N=1 | rpn=56 | mpi=56 | omp=2"        56              56                2               997962                17340                  16970285         

"N=2 | rpn=56 | mpi=112 | omp=2"        56             112                2               558735                 8908                   8364774          
"N=2 | rpn=2  | mpi=4  | omp=56"         2               4               56              1813151               193517                  17540511         
"N=2 | rpn=28 | mpi=56 | omp=4"        28              56                4               659602                15662                   8325181          
"N=2 | rpn=4  | mpi=8  | omp=28"         4               8               28              1042787                99132                   9263563         
"N=2 | rpn=14 | mpi=28 | omp=8"        14              28                8              1006469                29460                   8638316          
"N=2 | rpn=8  | mpi=16 | omp=14"         8              16               14               948077                50145                   8495852         

"N=4 | rpn=8  | mpi=32 | omp=14"         8              32               14               503320                25846                   4105777          
"N=4 | rpn=14 | mpi=56 | omp=8"        14              56                8               410120                15563                   4076145          
"N=4 | rpn=28 | mpi=112 | omp=4"        28             112                4               503603                 8356                   4122451         

"N=8 | rpn=2  | mpi=16 | omp=56"         2              16               56               541798                51349                   4355253          
"N=8 | rpn=56 | mpi=448 | omp=2"        56             448                2               344502                 2503                   2026740          
"N=8 | rpn=28 | mpi=224 | omp=4"        28             224                4               328780                 4449                   2014018          
"N=8 | rpn=4  | mpi=32 | omp=28"         4              32               28               563492                25513                   2140872          
"N=8 | rpn=8  | mpi=64 | omp=14"         8              64               14               303225                13558                   2042366          
"N=8 | rpn=14 | mpi=112 | omp=8"        14             112                8               314919                 7832                   2017075          

"N=10 | rpn=14 | mpi=140 | omp=8"        14             140                8               257211                 6336                   1669426          
"N=10 | rpn=8  | mpi=80 | omp=14"         8              80               14               389295                10658                   1687731          
"N=10 | rpn=4  | mpi=40 | omp=28"         4              40               28               666908                20729                   1766101          
"N=10 | rpn=28 | mpi=280 | omp=4"        28             280                4               270665                 3719                   1588536          
"N=10 | rpn=56 | mpi=560 | omp=2"        56             560                2               338686                 2076                   1582320          
"N=10 | rpn=2  | mpi=20 | omp=56"         2              20               56               606333                40161                   2322124               
```

![image.png](Parallel%20Programming%20Exam/image%201.png)

---

## Jacobi Solver

Hybrid + Parallel I/O in HDF5

```jsx
Grid Size = 5,000 * 5,000
Timer Precision = microseconds

Slurm Configuration
#!/bin/bash
#SBATCH -A ICT25_MHPC
#SBATCH -p dcgp_usr_prod
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4  # 56(2), 28(4), 14(8)
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=28
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --job-name=jacobi_pll_io
#SBATCH --output=logs/%x_%j.out

 
RANKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-8}
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-14}

export OMP_NUM_THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Compile Command
mpic++ src/main.cpp -std=c++20 -fopenmp \
  -I ../include -I"${HDF5_INCLUDE}" \
  -L"${HDF5_LIB}" -Wl,-rpath,"${HDF5_LIB}" \
  -lhdf5 \
  -o main.x
  
  
# Run Command
srun --cpu-bind=cores ./main.x "$INFILE"

Raw Data
Label                                             PRINT INTERVAL     COMM   INIT   MEMORY_IO    SOLVE_AND_SWAP
"N=1 | rpn=8  | omp=14 | print=500"                  500             5743200 24712   3753970         35425600
"N=1 | rpn=8  | omp=14 | print=2000"                 2000            5115560 24734   1143190         34541300
"N=1 | rpn=4  | omp=28 | print=500"                  500             4224980 47876   3929930         39566200
"N=1 | rpn=28 | omp=4  | print=500"                  500             8267750 8221    4554370         32796600
"N=1 | rpn=28 | omp=4  | print=2000"                 2000            6628190 8010    1314400         32766200
"N=1 | rpn=56 | omp=2  | print=500"                  500             6945380 4085    5188330         32190300
"N=1 | rpn=2  | omp=56 | print=500"                  500             1770240 94825   2265030         84954500
"N=2 | rpn=28 | omp=4  | print=500"                  500             5330280 3935    5592810         16283900
"N=2 | rpn=4  | omp=28 | print=500"                  500             3965330 24096   3924730         20317700
"N=2 | rpn=8  | omp=14 | print=500"                  500             5604450 12183   4041640         18487000
"N=2 | rpn=56 | omp=2  | print=500"                  500            41725700 2173    4552020         16248900
"N=2 | rpn=2  | omp=56 | print=500"                  500             2175970 47294   3668930         40010600
"N=8 | rpn=2  | omp=56 | print=500"                  500             1224500 12184   3059740          5126350
"N=8 | rpn=2  | omp=56 | print=2000"                 2000            1219690 11965   1991320          5162200
"N=8 | rpn=56 | omp=2  | print=500"                  500             1634830 617     6071380          3999250
"N=8 | rpn=56 | omp=2  | print=2000"                 2000            1627480 604     2151390          4004490
"N=8 | rpn=28 | omp=4  | print=500"                  500             1738320 1042    4222490          4054800
"N=8 | rpn=4  | omp=28 | print=500"                  500             1433600 6034    2972590          4693670
"N=8 | rpn=4  | omp=28 | print=1000"                 1000            1400260 6079    3746630          4721570
"N=8 | rpn=4  | omp=28 | print=2000"                 2000            1475220 6118    2051830          4741070
"N=8 | rpn=8  | omp=14 | print=500"                  500             1514660 3179    3155380          4359120
"N=8 | rpn=8  | omp=14 | print=1000"                 1000            1522490 3086    2555010          4369440
"N=8 | rpn=8  | omp=14 | print=2000"                 2000            1529130 3121    1410550          4383580
"N=10 | rpn=8  | omp=14 | print=500"                 500             1346030 2486    5374050         3502460
"N=10 | rpn=8  | omp=14 | print=1000"                1000            14097700 2526    3024230         3598440
"N=10 | rpn=8  | omp=14 | print=2000"                2000            1394770 2543    1471690         3510570
"N=10 | rpn=28 | omp=4  | print=500"                 500             1503180 839     6835130         3264290
"N=10 | rpn=28 | omp=4  | print=1000"                1000            1527310 835     3514370         3261020
"N=10 | rpn=28 | omp=4  | print=2000"                2000            1436960 832     1995070         3251500

```

![image.png](Parallel%20Programming%20Exam/image%202.png)

---

## FFTW

Using `fftw-3d` Implementation

```jsx
Timer Precision = Seconds
Mesh Size = 128 ^ 3
Time Step = 1e-4
Number of Steps = 500

#Slurm Config

#Compile Command
mpicc -O3 -Wall \
  diffusion.c plot_data.c fft_wrapper.c derivative.c \
  -lfftw3_mpi -lfftw3 -lm \
  -o diffusion.x

#Run Command
srun --cpu-bind=cores ./diffusion.x

#Raw Data
Nodes=1  Ranks/node=112
Slurm: ntasks-per-socket=56
[START] Tue Nov 11 18:04:21 2025
[END] Tue Nov 11 18:05:25 2025
time = 64000 micorseconds

Nodes=2  Ranks/node=112
Slurm: ntasks-per-socket=56
[START] Tue Nov 11 18:17:19 2025
[END] Tue Nov 11 18:17:45 2025
time = 25000 micorseconds

Nodes=8  Ranks/node=112
Slurm:   ntasks-per-socket=56
[START] Tue Nov 11 18:17:30 2025
[END] Tue Nov 11 18:17:42 2025
time = 12000 micorseconds

Nodes=10  Ranks/node=112
Slurm: ntasks-per-socket=56
[START] Tue Nov 11 18:19:37 2025
[END] Tue Nov 11 18:19:49 2025
12000 microseconds

Nodes=16  Ranks/node=112
Slurm:   ntasks-per-socket=56
[START] Tue Nov 11 18:19:36 2025
[END] Tue Nov 11 18:19:45 2025
11000 seconds

```

![image.png](Parallel%20Programming%20Exam/image%203.png)

<aside>
💡

But, I'm not sure of these results as on 13th Nov, I got very different results. Basically inverse of this.

I also noticed, that if I don’t delete the .dat files for concentration and diffusion, the time increases.

</aside>

Results from today

```jsx
Slurm Config
#SBATCH -A ICT25_MHPC
#SBATCH -p dcgp_usr_prod
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=112
#SBATCH --ntasks-per-socket=56
##SBATCH --cpus-per-task=1

#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00

#SBATCH --job-name=diffusion_fftw_mpi_custom
#SBATCH --output=logs/%x_%j.out

#Build Command
mpicc -O3 -Wall \
  diffusion.c plot_data.c fft_wrapper.c derivative.c \
  -lfftw3_mpi -lfftw3 -lm \
  -o diffusion.x
  
#Run Command
srun --cpu-bind=cores ./diffusion.x

#Raw Results
Nodes=1  Ranks/node=112
Slurm:  ntasks-per-socket=56
Grid Size = 1024^3, t_step = 1e-4, n_step = 50
[RUN] 2025-11-14 09:14:28 — launching with srun
slurmstepd: error: *** JOB 25692277 ON lrdn3699 CANCELLED AT 2025-11-14T09:27:48 DUE TO TIME LIMIT ***
srun: Job step aborted: Waiting up to 182 seconds for job step to finish.
slurmstepd: error: *** STEP 25692277.0 ON lrdn3699 CANCELLED AT 2025-11-14T09:27:48 DUE TO TIME LIMIT ***
/Exceeded 20 minutes/

Nodes=8  Ranks/node=112
Slurm:  ntasks-per-socket=56
Grid Size = 1024^3, t_step = 1e-4, n_step = 50
[START] Fri Nov 14 09:15:05 2025
 897 
  1 0.540321144570192 1.000000000000181 Elapsed time per iteration 14.743502 
 31 194779783215221387034624.000000000000000 1693953595668.457031250000000 Elapsed time per iteration 14.916941 
[END] Fri Nov 14 09:27:38 2025
12 minutes 33 seconds

Nodes=16  Ranks/node=112
Slurm:  ntasks-per-socket=56
Grid Size = 1024^3, t_step = 1e-4, n_step = 50
[START] Fri Nov 14 09:15:14 2025
 1793 
1 0.540321144570126 1.000000000000083 Elapsed time per iteration 7.971952 
31 194779784782612793393152.000000000000000 105876133032420.406250000000000 Elapsed time per iteration 8.204090 
[END] Fri Nov 14 09:22:09 2025
7 minutes 35 seconds

```