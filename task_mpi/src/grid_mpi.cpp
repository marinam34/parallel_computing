#include "grid_mpi.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "grid_mpi.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <vector>

void GridMPI::init(MPI_Comm comm, const Config& c){
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    Nx = c.Nx;
    Ny = c.Ny;
    Nz = c.Nz_internal;

    int dims[3] = {0, 0, 0};
    MPI_Dims_create(size, 3, dims);
    Px = dims[0];
    Py = dims[1];
    Pz = dims[2];


    int periods[3] = {1, 1, 0};
    int reorder = 0;      
    MPI_Comm cart_comm;

    MPI_Cart_create(comm, 3, dims, periods, reorder, &cart_comm);
    if (cart_comm == MPI_COMM_NULL) {
        throw std::runtime_error("MPI_Cart_create failed");
    }

    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);

    int coords[3];
    MPI_Cart_coords(cart_comm, cart_rank, 3, coords);
    rx = coords[0]; 
    ry = coords[1];  
    rz = coords[2];  

    struct Split {
        static int size_at(int N, int P, int rr){
            int base = N / P;
            int rem  = N % P;
            return base + (rr < rem ? 1 : 0);
        }
    };

    nx = Split::size_at(Nx, Px, rx);
    ny = Split::size_at(Ny, Py, ry);
    nz = Split::size_at(Nz, Pz, rz);

    MPI_Cart_shift(cart_comm, 0, 1, &nbr_xm, &nbr_xp);  
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_ym, &nbr_yp); 
    MPI_Cart_shift(cart_comm, 2, 1, &nbr_zm, &nbr_zp);  
    MPI_Comm_free(&cart_comm);
}

std::tuple<int,int> GridMPI::range_x() const {
    int beg = 0;
    int base = Nx / Px, rem = Nx % Px;
    for(int i = 0; i < rx; i++) beg += base + (i < rem ? 1 : 0);
    return std::make_tuple(beg, beg + nx - 1);
}

std::tuple<int,int> GridMPI::range_y() const {
    int beg = 0;
    int base = Ny / Py, rem = Ny % Py;
    for(int i = 0; i < ry; i++) beg += base + (i < rem ? 1 : 0);
    return std::make_tuple(beg, beg + ny - 1);
}

std::tuple<int,int> GridMPI::range_z() const {
    int beg = 1;
    int base = Nz / Pz, rem = Nz % Pz;
    for(int i = 0; i < rz; i++) beg += base + (i < rem ? 1 : 0);
    return std::make_tuple(beg, beg + nz - 1);
}

void GridMPI::exchange(double* u) const {
    MPI_Request reqs[12];
    int rcount = 0;

    auto pack_x_plane = [&](int i){
        std::vector<double> buf(ny * nz);
        int p = 0;
        for(int j = 1; j <= ny; j++)
            for(int k = 1; k <= nz; k++)
                buf[p++] = u[idx(i, j, k, nx, ny, nz)];
        return buf;
    };
    auto unpack_x_plane = [&](int i, const std::vector<double>& buf){
        int p = 0;
        for(int j = 1; j <= ny; j++)
            for(int k = 1; k <= nz; k++)
                u[idx(i, j, k, nx, ny, nz)] = buf[p++];
    };

    std::vector<double> recv_xm(ny * nz), recv_xp(ny * nz);
    std::vector<double> send_xm = pack_x_plane(1), send_xp = pack_x_plane(nx);
    MPI_Isend(send_xm.data(), (int)send_xm.size(), MPI_DOUBLE, nbr_xm, 100, MPI_COMM_WORLD, &reqs[rcount++]);
    MPI_Irecv(recv_xm.data(), (int)recv_xm.size(), MPI_DOUBLE, nbr_xm, 101, MPI_COMM_WORLD, &reqs[rcount++]);
    MPI_Isend(send_xp.data(), (int)send_xp.size(), MPI_DOUBLE, nbr_xp, 101, MPI_COMM_WORLD, &reqs[rcount++]);
    MPI_Irecv(recv_xp.data(), (int)recv_xp.size(), MPI_DOUBLE, nbr_xp, 100, MPI_COMM_WORLD, &reqs[rcount++]);

    auto pack_y_plane = [&](int j){
        std::vector<double> buf(nx * nz);
        int p = 0;
        for(int i = 1; i <= nx; i++)
            for(int k = 1; k <= nz; k++)
                buf[p++] = u[idx(i, j, k, nx, ny, nz)];
        return buf;
    };
    auto unpack_y_plane = [&](int j, const std::vector<double>& buf){
        int p = 0;
        for(int i = 1; i <= nx; i++)
            for(int k = 1; k <= nz; k++)
                u[idx(i, j, k, nx, ny, nz)] = buf[p++];
    };

    std::vector<double> recv_ym(nx * nz), recv_yp(nx * nz);
    std::vector<double> send_ym = pack_y_plane(1), send_yp = pack_y_plane(ny);
    MPI_Isend(send_ym.data(), (int)send_ym.size(), MPI_DOUBLE, nbr_ym, 200, MPI_COMM_WORLD, &reqs[rcount++]);
    MPI_Irecv(recv_ym.data(), (int)recv_ym.size(), MPI_DOUBLE, nbr_ym, 201, MPI_COMM_WORLD, &reqs[rcount++]);
    MPI_Isend(send_yp.data(), (int)send_yp.size(), MPI_DOUBLE, nbr_yp, 201, MPI_COMM_WORLD, &reqs[rcount++]);
    MPI_Irecv(recv_yp.data(), (int)recv_yp.size(), MPI_DOUBLE, nbr_yp, 200, MPI_COMM_WORLD, &reqs[rcount++]);

    auto pack_z_plane = [&](int k){
        std::vector<double> buf(nx * ny);
        int p = 0;
        for(int i = 1; i <= nx; i++)
            for(int j = 1; j <= ny; j++)
                buf[p++] = u[idx(i, j, k, nx, ny, nz)];
        return buf;
    };

    std::vector<double> recv_zm, recv_zp, send_zm, send_zp;
    if(nbr_zm != MPI_PROC_NULL){
        send_zm = pack_z_plane(1);
        recv_zm.resize(nx * ny);
        MPI_Isend(send_zm.data(), (int)send_zm.size(), MPI_DOUBLE, nbr_zm, 300, MPI_COMM_WORLD, &reqs[rcount++]);
        MPI_Irecv(recv_zm.data(), (int)recv_zm.size(), MPI_DOUBLE, nbr_zm, 301, MPI_COMM_WORLD, &reqs[rcount++]);
    }
    if(nbr_zp != MPI_PROC_NULL){
        send_zp = pack_z_plane(nz);
        recv_zp.resize(nx * ny);
        MPI_Isend(send_zp.data(), (int)send_zp.size(), MPI_DOUBLE, nbr_zp, 301, MPI_COMM_WORLD, &reqs[rcount++]);
        MPI_Irecv(recv_zp.data(), (int)recv_zp.size(), MPI_DOUBLE, nbr_zp, 300, MPI_COMM_WORLD, &reqs[rcount++]);
    }

    MPI_Waitall(rcount, reqs, MPI_STATUSES_IGNORE);

    unpack_x_plane(0, recv_xm);
    unpack_x_plane(nx + 1, recv_xp);
    unpack_y_plane(0, recv_ym);
    unpack_y_plane(ny + 1, recv_yp);

    if(nbr_zm != MPI_PROC_NULL){
        int p = 0;
        for(int i = 1; i <= nx; i++)
            for(int j = 1; j <= ny; j++)
                u[idx(i, j, 0, nx, ny, nz)] = recv_zm[p++];
    } else {
        for(int i = 1; i <= nx; i++)
            for(int j = 1; j <= ny; j++)
                u[idx(i, j, 0, nx, ny, nz)] = 0.0;
    }

    if(nbr_zp != MPI_PROC_NULL){
        int p = 0;
        for(int i = 1; i <= nx; i++)
            for(int j = 1; j <= ny; j++)
                u[idx(i, j, nz + 1, nx, ny, nz)] = recv_zp[p++];
    } else {
        for(int i = 1; i <= nx; i++)
            for(int j = 1; j <= ny; j++)
                u[idx(i, j, nz + 1, nx, ny, nz)] = 0.0;
    }
}

