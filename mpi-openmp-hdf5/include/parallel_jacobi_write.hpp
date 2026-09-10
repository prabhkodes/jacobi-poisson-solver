// parallel_write_minimal.hpp
#pragma once
#include <mpi.h>
#include <hdf5.h>
#include <cstdio>
#include <stdexcept>

inline void write_h5_parallel(
    const char* output,          // file path
    const char* dset_path,       //  dataset path per timestep, e.g. "/t0001"
    int NX, int NY,              // global dims: X (cols), Y (rows)
    int y_start,                 // first global row index owned by this rank
    int local_rows,              // number of *physical* rows this rank owns (no ghosts)
    int NG,                      // ghost rows on *top* of local buffer
    int ny_local_tot,            // total local buffer rows (ghosts included)
    int cols_local,              // local cols incl. halos (cols = N2+2)   
    const double* old_field     // pointer to start of local 2D buffer [ny_local_tot][NX]
    )              
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Info info = MPI_INFO_NULL;

    // File access with MPI-IO
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, info);

    hid_t file = H5Fcreate(output, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    // Added new

    H5Pclose(fapl);
    if (file < 0) {
        std::fprintf(stderr, "Rank %d: H5Fcreate failed for %s\n", rank, output);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    // Global dataspace [NY, NX]
    hsize_t gdim[2] = { (hsize_t)NY, (hsize_t)NX };
    hid_t filespace = H5Screate_simple(2, gdim, NULL);
    if (filespace < 0) {
        std::fprintf(stderr, "Rank %d: H5Screate_simple(filespace) failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    // This rank’s block (no ghosts)
    hsize_t start[2] = { (hsize_t)y_start, 0 };
    hsize_t count[2] = { (hsize_t)local_rows, (hsize_t)NX };

    // Guard overflow
    if (start[0] > (hsize_t)NY) count[0] = 0;
    if (start[0] + count[0] > (hsize_t)NY)
        count[0] = (hsize_t)NY - start[0];

    if (count[0] > 0)
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL);
    else
        H5Sselect_none(filespace);

    // Memory dataspace [ny_local_tot, NX]
    // hsize_t memdim[2] = { (hsize_t)ny_local_tot, (hsize_t)NX };
    hsize_t memdim[2] = { (hsize_t)ny_local_tot, (hsize_t)cols_local }; // use true stride
    // Added new

    hid_t memspace = H5Screate_simple(2, memdim, NULL);
    if (memspace < 0) {
        std::fprintf(stderr, "Rank %d: H5Screate_simple(memspace) failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    // Skip NG top ghost rows; write local_rows rows
    // hsize_t mem_start[2] = { (hsize_t)NG, 0};
    hsize_t mem_start[2] = { (hsize_t)NG, 1};


    if (count[0] > 0)
        H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_start, NULL, count, NULL);
    else
        H5Sselect_none(memspace);

    hid_t dset = H5Dcreate2(file, dset_path, H5T_NATIVE_DOUBLE,
                        filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    if (dset < 0) {
        std::fprintf(stderr, "Rank %d: H5Dcreate2 failed for %s\n", rank, dset_path);
        MPI_Abort(MPI_COMM_WORLD, 5);
    }

    // Collective write
    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

    herr_t rc = H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl, old_field);

    if (rc < 0) {
        std::fprintf(stderr,
            "Rank %d: H5Dwrite failed (startY=%llu, rows=%llu)\n",
            rank,
            (unsigned long long)start[0],
            (unsigned long long)count[0]);
        MPI_Abort(MPI_COMM_WORLD, 7);
    }

    // Cleanup
    H5Pclose(dxpl);
    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Dclose(dset);
    H5Fclose(file);
}





// Drop-in: new file each call (per step), dataset "/temperature", include halos
inline void write_h5_parallel_with_halo(
    const char* output,          // per-step file path, e.g. "files/T_0001.h5"
    const char* dset_path,       // should be "/temperature"
    int NX, int NY,              // global *interior* dims: X=N2, Y=n_global
    int y_start,                 // first global interior row owned by this rank (0-based)
    int local_rows,              // interior rows owned by this rank (N1)
    int NG,                      // top ghost rows in local buffer (usually 1)
    int ny_local_tot,            // local rows incl. halos (rows = N1+2)
    int cols_local,              // local cols incl. halos (cols = N2+2)
    const double* old_field      // base pointer to [ny_local_tot][cols_local]
) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- 1) File access (MPI-IO) & NEW file per step ---
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);


    hid_t file = H5Fcreate(output, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) {
        std::fprintf(stderr, "Rank %d: H5Fcreate failed for %s\n", rank, output);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    // --- 2) FILE dataspace: GLOBAL WITH HALOS [NY+2, NX+2] ---
    const hsize_t GX = (hsize_t)(NX + 2);  // width incl halos
    const hsize_t GY = (hsize_t)(NY + 2);  // height incl halos
    hsize_t gdim[2] = { GY, GX };
    hid_t filespace = H5Screate_simple(2, gdim, NULL);
    if (filespace < 0) {
        std::fprintf(stderr, "Rank %d: filespace create failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    // --- 3) Per-rank FILE selection (no overlap) ---
    const bool is_first = (rank == 0);
    const bool is_last  = (rank == size - 1);

    // Rank 0 writes [top halo + its interior], middle ranks write [their interior],
    // last rank writes [its interior + bottom halo]
    hsize_t file_start[2];
    hsize_t file_count[2];

    file_start[0] = (hsize_t)(is_first ? 0 : (y_start + 1));  // +1 to account for global top halo row
    file_start[1] = 0;                                        // include left halo
    file_count[0] = (hsize_t)(local_rows + (is_first ? 1 : 0) + (is_last ? 1 : 0));
    file_count[1] = GX;                                       // full width with halos

    if (file_start[0] + file_count[0] > GY) {
        // clip defensively (should be exact if inputs are consistent)
        file_count[0] = (file_start[0] < GY) ? (GY - file_start[0]) : 0;
    }

    if (file_count[0] > 0)
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, file_start, NULL, file_count, NULL);
    else
        H5Sselect_none(filespace);

    // --- 4) MEMORY dataspace: local buffer WITH halos [ny_local_tot, cols_local] ---
    hsize_t memdim[2] = { (hsize_t)ny_local_tot, (hsize_t)cols_local };
    hid_t memspace = H5Screate_simple(2, memdim, NULL);
    if (memspace < 0) {
        std::fprintf(stderr, "Rank %d: memspace create failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    // Match memory selection to file selection:
    //  - rank 0 starts at row 0 (includes its top halo), others at NG (their first interior row)
    //  - start col 0 to include left halo
    hsize_t mem_start[2];
    hsize_t mem_count[2];

    mem_start[0] = (hsize_t)(is_first ? 0 : NG);
    mem_start[1] = 0;
    mem_count[0] = file_count[0];  // same rows as file
    mem_count[1] = GX;             // NX+2 (full width with halos)


    if (mem_count[0] > 0)
        H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_start, NULL, mem_count, NULL);
    else
        H5Sselect_none(memspace);

    // --- 5) Create dataset (fixed name) with FILE shape [NY+2, NX+2] ---
    hid_t dset = H5Dcreate2(file, dset_path, H5T_NATIVE_DOUBLE,
                            filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) {
        std::fprintf(stderr, "Rank %d: H5Dcreate2 failed for %s\n", rank, dset_path);
        MPI_Abort(MPI_COMM_WORLD, 5);
    }

    // --- 6) Collective write ---
    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

    herr_t rc = H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl, old_field);
    if (rc < 0) {
        std::fprintf(stderr,
            "Rank %d: H5Dwrite failed (file_start={%llu,%llu} file_count={%llu,%llu})\n",
            rank,
            (unsigned long long)file_start[0], (unsigned long long)file_start[1],
            (unsigned long long)file_count[0], (unsigned long long)file_count[1]);
        MPI_Abort(MPI_COMM_WORLD, 7);
    }

    // --- 7) Cleanup ---
    H5Pclose(dxpl);
    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Dclose(dset);
    H5Fclose(file);
}
