#include "grid.hpp"


Grid::Grid(int Nx_, int Ny_, int Nz_,
           double Lx_, double Ly_, double Lz_)
    : Nx(Nx_), Ny(Ny_), Nz(Nz_),
      Lx(Lx_), Ly(Ly_), Lz(Lz_)
{

    hx = Lx / Nx;
    hy = Ly / Ny;
    hz = Lz / (Nz - 1);


    pitch_x = Nx;
    pitch_y = static_cast<std::size_t>(Ny) * pitch_x;
    total   = static_cast<std::size_t>(Nz) * pitch_y;


    u_prev.assign(total, 0.0);
    u_cur.assign(total, 0.0);
    u_next.assign(total, 0.0);
}


size_t Grid::idx(int i, int j, int k) const {
    return static_cast<std::size_t>(k) * pitch_y +
           static_cast<std::size_t>(j) * pitch_x +
           static_cast<std::size_t>(i);
}
