#include "solver_hybrid.hpp"
#include <cmath>
#include <iostream>
#include <tuple>
#include <omp.h>

SolverHybrid::SolverHybrid(const Config& cfg_, const GridMPI& grid_)
: cfg(cfg_), grid(grid_), an(cfg_)
{
    const size_t sz = (size_t)(grid.nx+2)*(grid.ny+2)*(grid.nz+2);
    u_nm1.assign(sz, 0.0);
    u_n.assign  (sz, 0.0);
    u_np1.assign(sz, 0.0);
}

void SolverHybrid::init_layers(){
    int ix0,ix1,iy0,iy1,iz0,iz1;
    std::tie(ix0,ix1)=grid.range_x();
    std::tie(iy0,iy1)=grid.range_y();
    std::tie(iz0,iz1)=grid.range_z();

    #pragma omp parallel for collapse(3) schedule(static)
    for(int li=1; li<=grid.nx; ++li){
        for(int lj=1; lj<=grid.ny; ++lj){
            for(int lk=1; lk<=grid.nz; ++lk){
                int gi = ix0 + (li-1);
                int gj = iy0 + (lj-1);
                int gk = iz0 + (lk-1);
                double x = gi*cfg.hx;
                double y = gj*cfg.hy;
                double z = gk*cfg.hz;
                u_n[I(li,lj,lk)] = an.phi(x,y,z);
            }
        }
    }

    grid.exchange(u_n.data());

    const double hx2=cfg.hx*cfg.hx, hy2=cfg.hy*cfg.hy, hz2=cfg.hz*cfg.hz;
    const double cxx = cfg.a*cfg.a*cfg.dt*cfg.dt/2.0;

    #pragma omp parallel for collapse(3) schedule(static)
    for(int i=1;i<=grid.nx;i++)
        for(int j=1;j<=grid.ny;j++)
            for(int k=1;k<=grid.nz;k++){
                double u0 = u_n[I(i,j,k)];
                double lap =
                    (u_n[I(i-1,j,k)] - 2*u0 + u_n[I(i+1,j,k)]) / hx2 +
                    (u_n[I(i,j-1,k)] - 2*u0 + u_n[I(i,j+1,k)]) / hy2 +
                    (u_n[I(i,j,k-1)] - 2*u0 + u_n[I(i,j,k+1)]) / hz2;
                u_np1[I(i,j,k)] = u0 + cxx*lap;
            }

    std::swap(u_nm1,u_n);
    std::swap(u_n,u_np1);
    grid.exchange(u_n.data());
}

void SolverHybrid::step(){
    const double hx2=cfg.hx*cfg.hx, hy2=cfg.hy*cfg.hy, hz2=cfg.hz*cfg.hz;
    const double a2dt2 = cfg.a*cfg.a*cfg.dt*cfg.dt;

    #pragma omp parallel for collapse(3) schedule(static)
    for(int i=1;i<=grid.nx;i++)
        for(int j=1;j<=grid.ny;j++)
            for(int k=1;k<=grid.nz;k++){
                double u0 = u_n[I(i,j,k)];
                double lap =
                    (u_n[I(i-1,j,k)] - 2*u0 + u_n[I(i+1,j,k)]) / hx2 +
                    (u_n[I(i,j-1,k)] - 2*u0 + u_n[I(i,j+1,k)]) / hy2 +
                    (u_n[I(i,j,k-1)] - 2*u0 + u_n[I(i,j,k+1)]) / hz2;
                u_np1[I(i,j,k)] = 2*u0 - u_nm1[I(i,j,k)] + a2dt2*lap;
            }

    std::swap(u_nm1,u_n);
    std::swap(u_n,u_np1);
    grid.exchange(u_n.data());
}

double SolverHybrid::compute_global_error(double t){
    int ix0,ix1,iy0,iy1,iz0,iz1;
    std::tie(ix0,ix1)=grid.range_x();
    std::tie(iy0,iy1)=grid.range_y();
    std::tie(iz0,iz1)=grid.range_z();

    double local_max = 0.0;

    #pragma omp parallel
    {
        double tmax = 0.0;
        #pragma omp for collapse(3) nowait schedule(static)
        for(int i=1;i<=grid.nx;i++){
            for(int j=1;j<=grid.ny;j++){
                for(int k=1;k<=grid.nz;k++){
                    int gi = ix0 + (i-1);
                    int gj = iy0 + (j-1);
                    int gk = iz0 + (k-1);
                    double x = gi * cfg.hx;
                    double y = gj * cfg.hy;
                    double z = gk * cfg.hz;
                    double err = std::abs(u_n[I(i,j,k)] - an.u(x,y,z,t));
                    if(err > tmax) tmax = err;
                }
            }
        }
        #pragma omp critical
        { if(tmax > local_max) local_max = tmax; }
    }

    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_max;
}

void SolverHybrid::run(){
    init_layers();
    const int steps = (int)std::ceil(cfg.T / cfg.dt);

    if(grid.rank==0){
        std::cout << "Hybrid MPI+OpenMP. Nx,Ny,Nz="<<cfg.Nx<<","<<cfg.Ny<<","<<cfg.Nz
                  << "  Lx,Ly,Lz="<<cfg.Lx<<","<<cfg.Ly<<","<<cfg.Lz
                  << "  a="<<cfg.a<<"  dt="<<cfg.dt<<"  steps="<<steps << "\n";
        std::cout.flush();
    }

    double t_total0 = MPI_Wtime();
    double t = cfg.dt;
    for(int n=1; n<=steps; ++n){
        double t0 = MPI_Wtime();
        step();
        t += cfg.dt;
        double err = compute_global_error(t);
        double t1 = MPI_Wtime();
        double dt_local = t1 - t0;
        double dt_step = 0.0;
        MPI_Allreduce(&dt_local, &dt_step, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(grid.rank==0){
            std::cout << "step " << n << "  t=" << t
                      << "  error_max=" << err
                      << "  step_time=" << dt_step << " s\n";
        }
    }
    double t_total1 = MPI_Wtime();
    double total = t_total1 - t_total0;
    double total_max = 0.0;
    MPI_Allreduce(&total, &total_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(grid.rank==0){
        std::cout << "TOTAL TIME: " << total_max << " s\n";
    }
}
