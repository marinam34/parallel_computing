cmake_minimum_required(VERSION 3.16)
project(task_hybrid CXX)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIBOMP_ROOT /opt/homebrew/opt/libomp)

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  find_library(OMP_LIB omp HINTS ${LIBOMP_ROOT}/lib REQUIRED)
  find_path(OMP_INCLUDE omp.h HINTS ${LIBOMP_ROOT}/include REQUIRED)

  add_executable(task_hybrid
    src/main_hybrid.cpp
    src/config.cpp
    src/analytic.cpp
    src/grid_mpi.cpp
    src/solver_hybrid.cpp
  )

  target_include_directories(task_hybrid PRIVATE include ${OMP_INCLUDE})
  target_compile_options(task_hybrid PRIVATE -Xpreprocessor -fopenmp)
  target_link_libraries(task_hybrid PRIVATE MPI::MPI_CXX ${OMP_LIB})
  target_link_options(task_hybrid PRIVATE -Wl,-rpath,${LIBOMP_ROOT}/lib)

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  find_library(OMP_LIB omp HINTS ${LIBOMP_ROOT}/lib REQUIRED)
  find_path(OMP_INCLUDE omp.h HINTS ${LIBOMP_ROOT}/include REQUIRED)

  add_executable(task_hybrid
    src/main_hybrid.cpp
    src/config.cpp
    src/analytic.cpp
    src/grid_mpi.cpp
    src/solver_hybrid.cpp
  )

  target_include_directories(task_hybrid PRIVATE include ${OMP_INCLUDE})
  target_compile_options(task_hybrid PRIVATE -fopenmp)
  target_link_libraries(task_hybrid PRIVATE MPI::MPI_CXX ${OMP_LIB})
  target_link_options(task_hybrid PRIVATE -Wl,-rpath,${LIBOMP_ROOT}/lib)

else()
  message(FATAL_ERROR "Unsupported compiler for macOS OpenMP flow")
endif()
