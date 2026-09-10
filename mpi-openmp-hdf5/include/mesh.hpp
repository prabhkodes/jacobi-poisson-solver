// mesh.hpp
#pragma once
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>   
#include <mpi.h>

#include "mpi_dt.hpp"
#include "printer.hpp"
#include "timer.hpp"
#include "parallel_jacobi_write.hpp"


inline void log_ts(const char* msg) {
    using namespace std::chrono;
    auto old_fill = std::cout.fill();          
    auto n  = system_clock::now();
    auto ms = duration_cast<milliseconds>(n.time_since_epoch()) % 1000;
    std::time_t tt = system_clock::to_time_t(n);

    std::cout << msg << " || "
              << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S")
              << '.' << std::setw(3) << std::setfill('0') << ms.count()
              << '\n';

    std::cout.fill(old_fill);                  
}


template <typename T>
class CMesh{    
public:
    long int N1, N2;              // local interior rows (N1) and global interior cols (N2 == N)  
    long int n_global;            // global interior size N                                       
    std::vector<T> old_field;     // size: (N1+2)*(N2+2)
    std::vector<T> new_field;     // size: (N1+2)*(N2+2)

    long max_steps{0};
    int print_interval{0};

    // MPI related
    int world_rank{0}, world_size{1};                                                             

    // these depend on MPI/world and N -> not const
    long base{0}, rem{0}, num_rows{0}, i_start{0}, i_end{0};    //elements split
    long cols{0}, rows{0}; // local cols and rows (N+2)

    T corner_value; // heat source
    
    CMesh(long int N, T corner, long steps, int p_interval);
    
    void apply_boundary();
    void exchange_boundaries();
    void jacobi_solver();

private:
    inline long idx(long i, long j) const noexcept { return i * cols + j; }
};


// ctor 
template <typename T>
CMesh<T>::CMesh(long int N, T corner, long steps, int p_interval) 
: N2(N), n_global(N), corner_value(corner), max_steps(steps), print_interval(p_interval)                                          
{   
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // divide and conquer baby
    base     = n_global / world_size;
    rem      = n_global %  world_size;
    num_rows = base + ((world_rank < rem) ? 1 : 0);
    i_start  = world_rank * base + std::min<long>(world_rank, rem);
    i_end    = i_start + num_rows;

    N1 = num_rows;                                                                               

    // Allocate local grid with halos
    rows = N1 + 2;       // add top/bottom halos
    cols = N2 + 2;       // add left/right halos

    const std::size_t size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);     

    {
    CTimer t("INIT fields with Halo");
    
    old_field.assign(size, T(0.5));
    new_field.assign(size, T(0.5));
    apply_boundary();
    }
    
}





template <typename T>
void CMesh<T>::apply_boundary() {
    /*
    Applies boundary as equal linear step on all corners
    Applies it to both old_field and new_field
    */

    // Top boundary row = 0 (only rank 0)
    if (world_rank == 0) {
        for (long j = 0; j < cols; ++j) {
            old_field[idx(0, j)] = T(0);
            new_field[idx(0, j)] = T(0);
        }
    }

    // Right boundary column = 0 (all ranks)
    for (long i = 0; i < rows; ++i) {
        old_field[idx(i, cols - 1)] = T(0);
        new_field[idx(i, cols - 1)] = T(0);
    }

    // Left boundary column for INTERIOR rows only (i = 1..N1), global-consistent ramp 0..corner
    for (long i = 1; i <= N1; ++i) {
        long G = i_start + i;  // 1..n_global
        T v = corner_value * static_cast<T>(G) / static_cast<T>(n_global);  // equal steps per row
        old_field[idx(i, 0)] = v;
        new_field[idx(i, 0)] = v;
    }


    // Bottom boundary row (only last rank): corner .. 0 across columns
    if (world_rank == world_size - 1) {
        // Bottom-left corner explicitly set to corner_value
        old_field[idx(rows - 1, 0)] = corner_value;
        new_field[idx(rows - 1, 0)] = corner_value;

        // Interior columns j=1..N2 ramp corner→0; right halo (j=cols-1) already 0
        for (long j = 1; j <= N2; ++j) {
            T v = corner_value * static_cast<T>(N2 - (j - 1)) / static_cast<T>(N2);
            old_field[idx(rows - 1, j)] = v;
            new_field[idx(rows - 1, j)] = v;
        }
    }
}





template <typename T>
void CMesh<T>::exchange_boundaries() {
    if (N1 == 0) return;  // no interior rows on this rank
    const int rank_above = (world_rank > 0) ? world_rank - 1 : MPI_PROC_NULL;
    const int rank_below = (world_rank + 1 < world_size) ? world_rank + 1 : MPI_PROC_NULL;

    const int count = static_cast<int>(N2);              // equals cols - 2
    T* send_up    = old_field.data() + idx(1,1);         // first interior row
    T* recv_up    = old_field.data() + idx(0, 1);        // top halo row
    T* send_down  = old_field.data() + idx(N1, 1);       // last interior row
    T* recv_down  = old_field.data() + idx(rows-1, 1);   // bottom halo row

    MPI_Datatype mpi_dt = get_mpi_datatype<T>();

    // Take care of rank above
    MPI_Sendrecv(send_up,   count, mpi_dt, rank_above,   100,
                 recv_up,   count, mpi_dt, rank_above,   200,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Take care of rank below
    MPI_Sendrecv(send_down, count, mpi_dt, rank_below, 200,
                 recv_down, count, mpi_dt, rank_below, 100,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

}



template <typename T>
void CMesh<T>::jacobi_solver() {
    if (rows <= 2 || cols <= 2 || N1 == 0 || max_steps <= 0) return;
    
    
    {
    CTimer t("Solve + swap + Communicate + Write");

    if (world_rank == 0) log_ts("STARTING solver execution at rank 0");

    const long i_last = rows - 1;
    const long j_last = cols - 1;

    long io_counter = 0;

    for (long step = 1; step <= max_steps; ++step) {
        {
        CTimer t("EXCHANGE BOUNDARIES");
        exchange_boundaries();

        }
        {
            CTimer t2("SOLVE AND SWAP");

            #pragma omp parallel for collapse(2) schedule(static)
            for (long i = 1; i < i_last; ++i) {
                for (long j = 1; j < j_last; ++j) {
                    new_field[idx(i, j)] =
                        (  old_field[idx(i-1, j)]
                         + old_field[idx(i,   j-1)]
                         + old_field[idx(i,   j+1)]
                         + old_field[idx(i+1, j)] ) * static_cast<T>(0.25);
                }
            }
            old_field.swap(new_field);
            
        }
        if (++io_counter == print_interval || step == max_steps)
        {
        CTimer t("Memory I/O");
        char fname[64];
        std::snprintf(fname, sizeof(fname), "files/T_%04ld.h5", step);
        const char* dset = "/temperature";

        write_h5_parallel_with_halo(
            fname, dset,
            (int)N2, (int)n_global,
            (int)i_start, (int)N1,
            /*NG=*/1,
            (int)rows, (int)cols,
            old_field.data()
        );
        io_counter = 0; // Reset Counter
        }
        
    } // 1st For loop
}
       
    if (world_rank == 0) log_ts("ENDING solver execution at rank 0");
}

