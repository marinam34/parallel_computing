cmake_minimum_required(VERSION 3.16)
project(task_mpi CXX)

find_package(MPI REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(task_mpi
  src/main_mpi.cpp
  src/config.cpp
  src/analytic.cpp
  src/grid_mpi.cpp
  src/solver_mpi.cpp
)

target_include_directories(task_mpi PRIVATE include)
target_link_libraries(task_mpi PRIVATE MPI::MPI_CXX)
