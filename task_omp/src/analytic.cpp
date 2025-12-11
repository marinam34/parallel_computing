#include "analytic.hpp"
#include <cmath>

double u_exact(double x, double y, double z, double t,
               double Lx, double Ly, double Lz, double omega)
{
    const double sx = std::sin(2.0 * M_PI * x / Lx);
    const double sy = std::sin(2.0 * M_PI * y / Ly);
    const double sz = std::sin(1.0 * M_PI * z / Lz);
    return sx * sy * sz * std::cos(omega * t);
}

void fill_analytic_layers(Grid& g, double tau, double omega)
{

    #pragma omp parallel for collapse(3)
    for (int k = 0; k < g.Nz; ++k)
      for (int j = 0; j < g.Ny; ++j)
        for (int i = 0; i < g.Nx; ++i) {
          const double x = i * g.hx, y = j * g.hy, z = k * g.hz;
          g.u_prev[g.idx(i,j,k)] = u_exact(x,y,z,0.0,  g.Lx,g.Ly,g.Lz, omega);
          g.u_cur [g.idx(i,j,k)] = u_exact(x,y,z,tau, g.Lx,g.Ly,g.Lz, omega);
        }


    #pragma omp parallel for collapse(2)
    for (int j=0;j<g.Ny;++j)
      for (int i=0;i<g.Nx;++i){
        g.u_prev[g.idx(i,j,0)]        = 0.0;
        g.u_prev[g.idx(i,j,g.Nz-1)]   = 0.0;
        g.u_cur [g.idx(i,j,0)]        = 0.0;
        g.u_cur [g.idx(i,j,g.Nz-1)]   = 0.0;
      }
}
