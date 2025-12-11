#pragma once
#include "config.hpp"

struct Analytic {
    explicit Analytic(const Config& c):cfg(c){}
    double u(double x,double y,double z,double t) const;
    double phi(double x,double y,double z) const { return u(x,y,z,0.0); }
private:
    const Config& cfg;
};
