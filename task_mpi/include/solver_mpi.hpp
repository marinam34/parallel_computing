#pragma once
#include "config.hpp"
#include "analytic.hpp"
#include "grid_mpi.hpp"
#include <vector>
#include <mpi.h>

struct SolverMPI {
    SolverMPI(const Config& cfg, const GridMPI& grid);

    void run();

private:
    const Config cfg;
    const GridMPI grid;
    Analytic an;

    std::vector<double> u_nm1, u_n, u_np1;
    void init_layers();
    void step();

    double compute_global_error(double t);

    inline size_t I(int i,int j,int k) const {
        return GridMPI::idx(i,j,k,grid.nx,grid.ny,grid.nz);
    }
};
