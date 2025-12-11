#pragma once
#include "grid.hpp"


double u_exact(double x, double y, double z, double t,
               double Lx, double Ly, double Lz, double omega);

void fill_analytic_layers(Grid& g, double tau, double omega);
