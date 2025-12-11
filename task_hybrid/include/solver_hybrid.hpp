#pragma once
#include "config.hpp"
#include "grid_mpi.hpp"
#include "analytic.hpp"
#include <vector>
#include <mpi.h>

struct SolverHybrid {
    const Config& cfg;
    const GridMPI& grid;
    Analytic an;

    std::vector<double> u_nm1, u_n, u_np1;

    explicit SolverHybrid(const Config& cfg_, const GridMPI& grid_);

    inline int I(int i,int j,int k) const {
        return (i*(grid.ny+2) + j)*(grid.nz+2) + k;
    }

    void init_layers();
    void step();
    double compute_global_error(double t);
    void run();
};
