#include "solver.hpp"
#include "analytic.hpp"
#include <cmath>
#include <omp.h>



static inline double laplace_aprox(const Grid& g, const std::vector<double>& u,
                                   int i, int j, int k)
{
    const int ixp = (i + 1 == g.Nx) ? 0 : (i + 1);
    const int ixm = (i == 0)        ? (g.Nx - 1) : (i - 1);
    const int jyp = (j + 1 == g.Ny) ? 0 : (j + 1);
    const int jym = (j == 0)        ? (g.Ny - 1) : (j - 1);

    const bool has_kp = (k + 1 < g.Nz);
    const bool has_km = (k > 0);

    const double invhx2 = 1.0 / (g.hx * g.hx);
    const double invhy2 = 1.0 / (g.hy * g.hy);
    const double invhz2 = 1.0 / (g.hz * g.hz);

    const double c   = u[g.idx(i, j, k)];
    const double ux  = u[g.idx(ixp, j, k)] - 2.0 * c + u[g.idx(ixm, j, k)];
    const double uy  = u[g.idx(i, jyp, k)] - 2.0 * c + u[g.idx(i, jym, k)];
    const double uzp = has_kp ? u[g.idx(i, j, k + 1)] : 0.0;
    const double uzm = has_km ? u[g.idx(i, j, k - 1)] : 0.0;
    const double uz  = uzp - 2.0 * c + uzm;

    return invhx2 * ux + invhy2 * uy + invhz2 * uz;
}

void Solver::start_layers(Grid& g)
{
    fill_analytic_layers(g, tau, omega);
}

StepStats Solver::step_seq_omp(Grid& g, double t_cur)
{
    const double a2_tau2 = (a * a) * (tau * tau);

    #pragma omp parallel for collapse(3)
    for (int k = 1; k < g.Nz - 1; ++k)
      for (int j = 0; j < g.Ny; ++j)
        for (int i = 0; i < g.Nx; ++i)
          g.u_next[g.idx(i, j, k)] =
              2.0 * g.u_cur[g.idx(i, j, k)]
            -       g.u_prev[g.idx(i, j, k)]
            + a2_tau2 * laplace_aprox(g, g.u_cur, i, j, k);

    #pragma omp parallel for collapse(2)
    for (int j = 0; j < g.Ny; ++j)
      for (int i = 0; i < g.Nx; ++i) {
        g.u_next[g.idx(i, j, 0)]        = 0.0;
        g.u_next[g.idx(i, j, g.Nz - 1)] = 0.0;
      }

    const double t_next = t_cur + tau;
    double err = 0.0;

    #pragma omp parallel for collapse(3) reduction(max:err)
    for (int k = 0; k < g.Nz; ++k)
      for (int j = 0; j < g.Ny; ++j)
        for (int i = 0; i < g.Nx; ++i) {
          const double x=i*g.hx, y=j*g.hy, z=k*g.hz;
          const double ua = u_exact(x,y,z,t_next, g.Lx,g.Ly,g.Lz, omega);
          const double e  = std::fabs(g.u_next[g.idx(i,j,k)] - ua);
          if (e > err) err = e;
        }

    g.u_prev.swap(g.u_cur);
    g.u_cur.swap(g.u_next);

    StepStats stats;
    stats.err_max = err;
    return stats;
}
