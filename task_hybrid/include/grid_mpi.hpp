#pragma once
#include "config.hpp"
#include <mpi.h>
#include <vector>
#include <tuple>

struct GridMPI {
    int rank=0, size=1;
    int Px=1, Py=1, Pz=1;
    int rx=0, ry=0, rz=0; 
    int nx=0, ny=0, nz=0;
    int Nx=0, Ny=0, Nz=0;

    int nbr_xm=-1, nbr_xp=-1, nbr_ym=-1, nbr_yp=-1, nbr_zm=-1, nbr_zp=-1;

    static inline size_t idx(int i,int j,int k,int nx,int ny,int nz){
        return static_cast<size_t>( ( (i)*(ny+2) + j )*(nz+2) + k );
    }

    void init(MPI_Comm comm, const Config& c);
    std::tuple<int,int> range_x() const;
    std::tuple<int,int> range_y() const;
    std::tuple<int,int> range_z() const;

    void exchange(double* u) const;
};
