#pragma once
#include <vector>
#include <cstddef>

class Grid {
public:
    int Nx, Ny, Nz;
    double Lx, Ly, Lz;
    double hx, hy, hz;
    std::size_t pitch_x, pitch_y, total;

    std::vector<double> u_prev, u_cur, u_next;

    Grid(int Nx_, int Ny_, int Nz_,
         double Lx_, double Ly_, double Lz_);

    std::size_t idx(int i, int j, int k) const;
};
