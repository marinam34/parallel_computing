#pragma once
#include "config.hpp"
#include "timing.hpp"
#include "grid_mpi.hpp"
#include "analytic.hpp"
#include <vector>
#include <mpi.h>

struct SolverCUDA {
    const Config& cfg;
    const GridMPI& grid;
    Analytic an;
    Timer timer;

    std::vector<double> h_u_nm1, h_u_n, h_u_np1;

    double *d_send_xm = nullptr, *d_recv_xm = nullptr;
    double *d_send_xp = nullptr, *d_recv_xp = nullptr;
    double *d_send_ym = nullptr, *d_recv_ym = nullptr;
    double *d_send_yp = nullptr, *d_recv_yp = nullptr;
    double *d_send_zm = nullptr, *d_recv_zm = nullptr;
    double *d_send_zp = nullptr, *d_recv_zp = nullptr;

    std::vector<double> h_send_xm, h_recv_xm;
    std::vector<double> h_send_xp, h_recv_xp;
    std::vector<double> h_send_ym, h_recv_ym;
    std::vector<double> h_send_yp, h_recv_yp;
    std::vector<double> h_send_zm, h_recv_zm;
    std::vector<double> h_send_zp, h_recv_zp;

    double* d_u_nm1 = nullptr;
    double* d_u_n   = nullptr;
    double* d_u_np1 = nullptr;

    double* d_block_max_error = nullptr; 
    
    double cxx = 0.0, cyy = 0.0, czz = 0.0;

    explicit SolverCUDA(const Config& cfg_, const GridMPI& grid_);
    ~SolverCUDA();

    inline size_t I(int i,int j,int k) const {
        return static_cast<size_t>((i*(grid.ny+2) + j)*(grid.nz+2) + k);
    }

    void init_layers();              
    void step();   
    double compute_global_error(double t); 
    void run(); 
    
    void allocate_halos();
    void free_halos(); 
};
