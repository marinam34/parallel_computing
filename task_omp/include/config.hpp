#pragma once
#include <string>

struct Config {
    double Lx = 1.0, Ly = 1.0, Lz = 1.0;

    int Nx = 64, Ny = 64, Nz = 64;

    double T = 0.1;
    double cfl = 0.95;

    double hx = 0, hy = 0, hz = 0;
    double dt = 0, a = 0, omega = 0;


    static Config parse_cmd(int argc, char** argv);
};
