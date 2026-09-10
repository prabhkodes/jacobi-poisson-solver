// main.cpp 
#include <string>
#include <mpi.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <filesystem>

#include "../include/mesh.hpp"
#include "../include/timer.hpp"
#include "../include/printer.hpp"


int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int world_rank = 0, world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // initialise input parameters
  long int N = 0;
  long int max_steps = 0;
  double corner = 0.0;
  long double field_fill = 0.0;

  // flags to check if values have been parsed from input file
  bool have_N = false, have_steps = false, have_corner = false, have_fill = false; 

  if (argc != 2) {
      std::cerr << "ERROR: pass exactly one argument: jacobian.in\n";
      return 1;
    }

    std::ifstream input(argv[1]);
    if (!input) {
      std::cerr << "ERROR: cannot open input file\n";
      return 1;
    }


    std::string line, key;
    while (std::getline(input, line)) {
        std::stringstream ss(line);
        ss >> key;
        if (key == "dimension_nb") { ss >> N;           have_N = true; } // size of mesh
        else if (key == "corner")  { ss >> corner;      have_corner = true; } // heat source 
        else if (key == "field_fill") { ss >> field_fill; have_fill = true; } // initial fill values
        else if (key == "max_steps")  { ss >> max_steps;  have_steps = true; } // max iterations
        // ignore anything else 
      }

    if (!(have_N && have_corner && have_fill && have_steps)) { //check required params
        std::cerr << "ERROR: missing required params (dimension_nb, corner, field_fill, max_steps)\n";
        return 1;
      }
  
    
    // Create local grid and apply boundary condition
    CMesh<double> mesh_local(/*N2=*/N, /*corner=*/corner, /*max_steps=*/max_steps);
     
    // ===============================================================================
    // ---- PRINT LOCAL GRIDS PER RANK (ordered) ----

    // std::cout << "This is initial mesh, before solving" << std::endl;

    // for (int r = 0; r < world_size; ++r) {
    //     if (world_rank == r) {
    //         std::cout << "\n===== Rank " << world_rank
    //                   << " (num_rows=" << mesh_local.N1
    //                   << ", N=" << mesh_local.N2 << ") =====\n";

    //         std::cout << "[old_field]\n";
    //         utils::print_grid_local<double>(mesh_local.old_field,
    //                                  /*num_rows=*/mesh_local.N1,
    //                                  /*N=*/mesh_local.N2,
    //                                  /*precision=*/1, /*width=*/6);

    //         std::cout << "\n[new_field]\n";
    //         utils::print_grid_local<double>(mesh_local.new_field,
    //                                  /*num_rows=*/mesh_local.N1,
    //                                  /*N=*/mesh_local.N2,
    //                                  /*precision=*/1, /*width=*/6);
    //         std::cout.flush();
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }
    // ---- PRINT LOCAL GRIDS PER RANK (ordered) ----
    // ===============================================================================


    mesh_local.jacobi_solver(); 

    // Gather times from all processes and write to statistics.txt
    std::vector<TimerData> all_timings;
    CTimer::gather_and_print(0, all_timings);

    // ===============================================================================
    // ---- PRINT LOCAL GRIDS PER RANK (ordered) ----

    std::cout << "This is the final mesh, AFTER solving" << std::endl;

    for (int r = 0; r < world_size; ++r) {
        if (world_rank == r) {
            std::cout << "\n===== Rank " << world_rank
                      << " (num_rows=" << mesh_local.N1
                      << ", N=" << mesh_local.N2 << ") =====\n";

            std::cout << "[old_field]\n";
            utils::print_grid_local<double>(mesh_local.old_field,
                                     /*num_rows=*/mesh_local.N1,
                                     /*N=*/mesh_local.N2,
                                     /*precision=*/1, /*width=*/6);

            std::cout << "\n[new_field]\n";
            utils::print_grid_local<double>(mesh_local.new_field,
                                     /*num_rows=*/mesh_local.N1,
                                     /*N=*/mesh_local.N2,
                                     /*precision=*/1, /*width=*/6);
            std::cout.flush();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    // ---- PRINT LOCAL GRIDS PER RANK (ordered) ----
    // ===============================================================================


    MPI_Finalize();

}
