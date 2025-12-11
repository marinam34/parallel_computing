#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <omp.h>

#include "config.hpp"
#include "grid.hpp"
#include "solver.hpp"
#include "analytic.hpp"

int main(int argc, char** argv) {
    Config cfg = Config::parse_cmd(argc, argv);

    Grid g(cfg.Nx, cfg.Ny, cfg.Nz, cfg.Lx, cfg.Ly, cfg.Lz);

    const int steps = std::max(1, (int)std::ceil(cfg.T / cfg.dt));
    const double tau = cfg.dt;

    int nthreads = 1;
    #pragma omp parallel
    {
        #pragma omp master
        nthreads = omp_get_num_threads();
    }

    std::cout << std::fixed;
    std::cout << "parameters:\n"
              << "       a       = " << std::setprecision(12) << cfg.a   << "\n"
              << "       tau     = " << std::setprecision(12) << cfg.dt  << "\n"
              << "       CFL     = " << std::setprecision(6)  << cfg.cfl << "\n"
              << "       grid    = " << cfg.Nx << "x" << cfg.Ny << "x" << cfg.Nz << "\n"
              << "       T_req   = " << std::setprecision(6)  << cfg.T   << "\n"
              << "       steps   = " << steps << "\n"
              << "   OpenMP threads = " << nthreads << "\n";

    Solver solver(tau, cfg.a, cfg.omega);

    const double t0 = omp_get_wtime();

    solver.start_layers(g);

    double t_cur = tau; 
    double final_err = 0.0;

    std::cout << "--------------------------------------\n";
    for (int n = 2; n <= steps; ++n) {
        StepStats st = solver.step_seq_omp(g, t_cur);
        t_cur += tau;
        final_err = st.err_max;

        std::cout << " step = " << std::setw(4) << n
                  << "   t = " << std::setw(10) << std::fixed << std::setprecision(6) << t_cur
                  << "   err_max = " << std::scientific << std::setprecision(3) << st.err_max
                  << "\n";
    }

    const double t1 = omp_get_wtime();
    std::cout << std::fixed << std::setprecision(6)
              << "--------------------------------------\n"
              << "Final time = " << t_cur
              << "   Final error = " << std::scientific << std::setprecision(6) << final_err << "\n"
              << std::fixed << std::setprecision(3)
              << "Total runtime: " << (t1 - t0) << " s"
              << " (" << (t1 - t0) * 1000.0 << " ms)\n";

    return 0;
}
