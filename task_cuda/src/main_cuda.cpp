#include "config.hpp"
#include "grid_mpi.hpp"
#include "solver_cuda.hpp"
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    Config cfg = Config::parse_cmd(argc, argv);
    if(rank==0){
        std::cout << "MPI + CUDA run\n";
    }

    GridMPI grid;
    grid.init(MPI_COMM_WORLD, cfg);

    SolverCUDA solver(cfg, grid);
    solver.run();

    MPI_Finalize();
    return 0;
}
