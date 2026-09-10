#pragma once
#include <type_traits>
#include <cstdint>
#include <mpi.h>


// Templated inline function to get the corresponding
// MPI_Datatype from c++ primitive data-types

// Falls back to MPI_BYTE

template <typename T>
inline MPI_Datatype get_mpi_datatype() {
    static_assert(std::is_trivially_copyable_v<T>,
                  "T must be trivially copyable to be sent with raw MPI.");

    if constexpr (std::is_same_v<T, char>)              return MPI_CHAR;
    else if constexpr (std::is_same_v<T, signed char>)  return MPI_SIGNED_CHAR;
    else if constexpr (std::is_same_v<T, unsigned char>)return MPI_UNSIGNED_CHAR;
    else if constexpr (std::is_same_v<T, short>)         return MPI_SHORT;
    else if constexpr (std::is_same_v<T, unsigned short>)return MPI_UNSIGNED_SHORT;
    else if constexpr (std::is_same_v<T, int>)           return MPI_INT;
    else if constexpr (std::is_same_v<T, unsigned int>)  return MPI_UNSIGNED;
    else if constexpr (std::is_same_v<T, long>)          return MPI_LONG;
    else if constexpr (std::is_same_v<T, unsigned long>) return MPI_UNSIGNED_LONG;
    else if constexpr (std::is_same_v<T, long long>)     return MPI_LONG_LONG_INT;
    else if constexpr (std::is_same_v<T, unsigned long long>) return MPI_UNSIGNED_LONG_LONG;
    else if constexpr (std::is_same_v<T, float>)         return MPI_FLOAT;
    else if constexpr (std::is_same_v<T, double>)        return MPI_DOUBLE;
    else if constexpr (std::is_same_v<T, long double>)   return MPI_LONG_DOUBLE;
    else if constexpr (std::is_same_v<T, std::int32_t>)  return MPI_INT32_T;
    else if constexpr (std::is_same_v<T, std::int64_t>)  return MPI_INT64_T;
    else if constexpr (std::is_same_v<T, std::uint32_t>) return MPI_UINT32_T;
    else if constexpr (std::is_same_v<T, std::uint64_t>) return MPI_UINT64_T;
    else                                                 return MPI_BYTE; // generic fallback
}
