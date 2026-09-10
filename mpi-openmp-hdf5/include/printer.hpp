// printer.hpp
#pragma once
#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>
#include <filesystem>
#include <mpi.h>

// Forward declare CMesh to avoid including mesh.hpp
template <typename T> class CMesh;

namespace utils {

// Build base name like ../files/jac_0000
template <typename T>
void print_namefile(std::ostream& os, const T& val){
    os << "files/jac_" << std::setfill('0') << std::setw(4) << val;
}

// Each rank writes its local grid to ../files/jac_0000_r<rank>.dat
template <typename T>
void print_grid(const CMesh<T>& m, const int& val){
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::ostringstream fname;
    print_namefile(fname, val);
    fname << "_r" << rank << ".dat";

    std::filesystem::create_directories("files");

    std::ofstream out(fname.str());
    if (!out) {
        std::cerr << "ERROR: cannot open " << fname.str() << " for writing\n";
        return;
    }

    for (long i = 0; i < m.rows; ++i) {
        for (long j = 0; j < m.cols; ++j) {
            out << std::fixed << std::setprecision(6)
                << m.old_field[i * m.cols + j]
                << (j + 1 < m.cols ? ' ' : '\n');
        }
    }
}


template <typename T>
void print_grid_local(const std::vector<T>& vec,
                      long num_rows,      // local interior rows on this rank
                      long N,             // global interior cols
                      int precision = 2,
                      int width = 7)
{
    const long rows = num_rows + 2;  // local with halos
    const long cols = N + 2;         // global with halos
    auto idx = [cols](long i, long j){ return i * cols + j; };

    auto clean = [](T x){
        if constexpr (std::is_floating_point_v<T>)
            return (std::abs(x) < static_cast<T>(1e-12)) ? T(0) : x;
        return x;
    };

    std::cout << std::fixed << std::setprecision(precision);
    for (long i = 0; i < rows; ++i) {
        for (long j = 0; j < cols; ++j) {
            std::cout << std::setw(width) << clean(vec[idx(i,j)]) << ' ';
        }
        std::cout << '\n';
    }
}



  template <typename T>
  void print_grid_interior(const std::vector<T>& vec, int N,
                          int precision = 2, int width = 7)
  {
      const int dim  = N + 2;
      auto idx = [dim](int i, int j){ return i * dim + j; };

      auto clean = [](T x){
          if (std::is_floating_point_v<T>) {
              if (std::abs(x) < static_cast<T>(1e-12)) return T(0);
          }
          return x;
      };

      std::cout << std::fixed << std::setprecision(precision);
      for (int i = 1; i <= N; ++i) {
          for (int j = 1; j <= N; ++j) {
              std::cout << std::setw(width) << clean(vec[idx(i,j)]) << ' ';
          }
          std::cout << '\n';
      }
  }
} // namespace utils

