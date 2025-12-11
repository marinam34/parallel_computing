#include "analytic.hpp"
#include <cmath>

double Analytic::u(double x, double y, double z, double t) const {
    const double sx = std::sin(2.0*M_PI*x/cfg.Lx + 3.0*M_PI);
    const double sy = std::sin(2.0*M_PI*y/cfg.Ly + 2.0*M_PI);
    const double sz = std::sin(1.0*M_PI*z/cfg.Lz);
    return sx * sy * sz * std::cos(cfg.omega*t + M_PI);
}
