#include "solver_cuda.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

static void cuda_check(cudaError_t err, const char* msg){
    if(err != cudaSuccess){
        std::cerr << "CUDA error at " << msg << ": "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA failure");
    }
}


__global__
void wave_step_opt_kernel(const double* __restrict__ u_nm1,
                          const double* __restrict__ u_n,
                          double* __restrict__ u_np1,
                          int nx, int ny, int nz,
                          double cxx, double cyy, double czz)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y + 1;
    int i = blockIdx.z + 1;

    if(k > nz) return;

    int ny2 = ny + 2;
    int nz2 = nz + 2;

    size_t idx = (static_cast<size_t>(i) * ny2 + j) * nz2 + k;

    size_t idx_im = (static_cast<size_t>(i-1) * ny2 + j) * nz2 + k;
    size_t idx_ip = (static_cast<size_t>(i+1) * ny2 + j) * nz2 + k;
    size_t idx_jm = (static_cast<size_t>(i) * ny2 + (j-1)) * nz2 + k;
    size_t idx_jp = (static_cast<size_t>(i) * ny2 + (j+1)) * nz2 + k;
    size_t idx_km = idx - 1;
    size_t idx_kp = idx + 1;

    double u0 = u_n[idx];

    double lap =
        (u_n[idx_im] - 2.0 * u0 + u_n[idx_ip]) * cxx +
        (u_n[idx_jm] - 2.0 * u0 + u_n[idx_jp]) * cyy +
        (u_n[idx_km] - 2.0 * u0 + u_n[idx_kp]) * czz;

    u_np1[idx] = 2.0 * u0 - u_nm1[idx] + lap;
}

__global__ void pack_x_kernel(const double* u, double* buf, int i_plane, int ny, int nz, int ny2, int nz2){
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if(k > nz || j > ny) return;
    int buf_idx = (j-1) * nz + (k-1);
    size_t grid_idx = (static_cast<size_t>(i_plane) * ny2 + j) * nz2 + k;
    buf[buf_idx] = u[grid_idx];
}

__global__ void unpack_x_kernel(double* u, const double* buf, int i_plane, int ny, int nz, int ny2, int nz2){
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if(k > nz || j > ny) return;

    int buf_idx = (j-1) * nz + (k-1);
    size_t grid_idx = (static_cast<size_t>(i_plane) * ny2 + j) * nz2 + k;

    u[grid_idx] = buf[buf_idx];
}


__global__ void pack_y_kernel(const double* u, double* buf, int j_plane, int nx, int nz, int ny2, int nz2){
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if(k > nz || i > nx) return;


    int buf_idx = (i-1) * nz + (k-1);
    
    size_t grid_idx = (static_cast<size_t>(i) * ny2 + j_plane) * nz2 + k;
    buf[buf_idx] = u[grid_idx];
}

__global__ void unpack_y_kernel(double* u, const double* buf, int j_plane, int nx, int nz, int ny2, int nz2){
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if(k > nz || i > nx) return;

    int buf_idx = (i-1) * nz + (k-1);
    size_t grid_idx = (static_cast<size_t>(i) * ny2 + j_plane) * nz2 + k;
    u[grid_idx] = buf[buf_idx];
}

__global__ void pack_z_kernel(const double* u, double* buf, int k_plane, int nx, int ny, int ny2, int nz2){
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if(j > ny || i > nx) return;

    int buf_idx = (i-1) * ny + (j-1);
    
    size_t grid_idx = (static_cast<size_t>(i) * ny2 + j) * nz2 + k_plane;
    buf[buf_idx] = u[grid_idx];
}

__global__ void unpack_z_kernel(double* u, const double* buf, int k_plane, int nx, int ny, int ny2, int nz2){
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if(j > ny || i > nx) return;

    int buf_idx = (i-1) * ny + (j-1);
    size_t grid_idx = (static_cast<size_t>(i) * ny2 + j) * nz2 + k_plane;
    u[grid_idx] = buf[buf_idx];
}


__global__ void error_kernel(const double* u_n, double* block_results,
                             int nx, int ny, int nz, 
                             double hx, double hy, double hz,
                             int i0, int j0, int k0, 
                             double t,
                             double Lx, double Ly, double Lz,
                             double omega)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx_lin = bid * blockDim.x + tid;
    
    int total_points = nx * ny * nz;
    
    double thread_max = 0.0;

    int ny2 = ny + 2;
    int nz2 = nz + 2;

    for(int idx = idx_lin; idx < total_points; idx += gridDim.x * blockDim.x){
        int k_minus_1 = idx % nz;
        int rest = idx / nz;
        int j_minus_1 = rest % ny;
        int i_minus_1 = rest / ny;

        int i = i_minus_1 + 1;
        int j = j_minus_1 + 1;
        int k = k_minus_1 + 1;

        double x = (i0 + i - 1) * hx;
        double y = (j0 + j - 1) * hy;
        double z = (k0 + k - 1) * hz;

       
        double kx = 2.0 * M_PI / Lx;
        double ky = 2.0 * M_PI / Ly;
        double kz = 1.0 * M_PI / Lz;
        
        double val = sin(kx * x) * sin(ky * y) * sin(kz * z) * cos(omega * t);

        size_t grid_idx = (static_cast<size_t>(i) * ny2 + j) * nz2 + k;
        double diff = fabs(u_n[grid_idx] - val);
        if(diff > thread_max) thread_max = diff;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        double other = __shfl_down_sync(0xFFFFFFFF, thread_max, offset);
        if (other > thread_max) thread_max = other; 
    }
     
    if ((tid & 31) == 0) {
        int warp_id = tid / 32;
        int global_warp_id = bid * (blockDim.x / 32) + warp_id;
        block_results[global_warp_id] = thread_max;
    }
}


SolverCUDA::SolverCUDA(const Config& cfg_, const GridMPI& grid_)
: cfg(cfg_), grid(grid_), an(cfg_)
{
    const size_t sz = static_cast<size_t>(grid.nx+2)
                    * static_cast<size_t>(grid.ny+2)
                    * static_cast<size_t>(grid.nz+2);

    h_u_nm1.assign(sz, 0.0);
    h_u_n.assign  (sz, 0.0);
    h_u_np1.assign(sz, 0.0);

    double alpha = cfg.a * cfg.dt;
    cxx = (alpha * alpha) / (cfg.hx * cfg.hx);
    cyy = (alpha * alpha) / (cfg.hy * cfg.hy);
    czz = (alpha * alpha) / (cfg.hz * cfg.hz);

    cuda_check(cudaMalloc(&d_u_nm1, sz * sizeof(double)), "cudaMalloc d_u_nm1");
    cuda_check(cudaMalloc(&d_u_n,   sz * sizeof(double)), "cudaMalloc d_u_n");
    cuda_check(cudaMalloc(&d_u_np1, sz * sizeof(double)), "cudaMalloc d_u_np1");

    allocate_halos();
    
    cuda_check(cudaMalloc(&d_block_max_error, sizeof(double)), "cudaMalloc reduction");
}

SolverCUDA::~SolverCUDA(){
    if(d_u_nm1) cudaFree(d_u_nm1);
    if(d_u_n)   cudaFree(d_u_n);
    if(d_u_np1) cudaFree(d_u_np1);

    free_halos();
    if(d_block_max_error) cudaFree(d_block_max_error);
}



void SolverCUDA::allocate_halos(){
    int nx = grid.nx;
    int ny = grid.ny;
    int nz = grid.nz;

    size_t sz_x = (size_t)ny * nz * sizeof(double);
    cuda_check(cudaMalloc(&d_send_xm, sz_x), "Halo X");
    cuda_check(cudaMalloc(&d_recv_xm, sz_x), "Halo X");
    cuda_check(cudaMalloc(&d_send_xp, sz_x), "Halo X");
    cuda_check(cudaMalloc(&d_recv_xp, sz_x), "Halo X");

    h_send_xm.resize(ny*nz); h_recv_xm.resize(ny*nz);
    h_send_xp.resize(ny*nz); h_recv_xp.resize(ny*nz);

    size_t sz_y = (size_t)nx * nz * sizeof(double);
    cuda_check(cudaMalloc(&d_send_ym, sz_y), "Halo Y");
    cuda_check(cudaMalloc(&d_recv_ym, sz_y), "Halo Y");
    cuda_check(cudaMalloc(&d_send_yp, sz_y), "Halo Y");
    cuda_check(cudaMalloc(&d_recv_yp, sz_y), "Halo Y");

    h_send_ym.resize(nx*nz); h_recv_ym.resize(nx*nz);
    h_send_yp.resize(nx*nz); h_recv_yp.resize(nx*nz);

    size_t sz_z = (size_t)nx * ny * sizeof(double);
    cuda_check(cudaMalloc(&d_send_zm, sz_z), "Halo Z");
    cuda_check(cudaMalloc(&d_recv_zm, sz_z), "Halo Z");
    cuda_check(cudaMalloc(&d_send_zp, sz_z), "Halo Z");
    cuda_check(cudaMalloc(&d_recv_zp, sz_z), "Halo Z");

    h_send_zm.resize(nx*ny); h_recv_zm.resize(nx*ny);
    h_send_zp.resize(nx*ny); h_recv_zp.resize(nx*ny);
}

void SolverCUDA::free_halos(){
    if(d_send_xm) cudaFree(d_send_xm);
    if(d_recv_xm) cudaFree(d_recv_xm);
    if(d_send_xp) cudaFree(d_send_xp);
    if(d_recv_xp) cudaFree(d_recv_xp);

    if(d_send_ym) cudaFree(d_send_ym);
    if(d_recv_ym) cudaFree(d_recv_ym);
    if(d_send_yp) cudaFree(d_send_yp);
    if(d_recv_yp) cudaFree(d_recv_yp);

    if(d_send_zm) cudaFree(d_send_zm);
    if(d_recv_zm) cudaFree(d_recv_zm);
    if(d_send_zp) cudaFree(d_send_zp);
    if(d_recv_zp) cudaFree(d_recv_zp);
}

void SolverCUDA::init_layers(){
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int nz = grid.nz;

    const double hx = cfg.hx;
    const double hy = cfg.hy;
    const double hz = cfg.hz;

    auto xr = grid.range_x();
    auto yr = grid.range_y();
    auto zr = grid.range_z();

    int i0 = std::get<0>(xr);
    int j0 = std::get<0>(yr);
    int k0 = std::get<0>(zr);
    
    double t0 = MPI_Wtime();
    #pragma omp parallel for collapse(3)
    for(int i = 1; i <= nx; ++i){
        for(int j = 1; j <= ny; ++j){
            for(int k = 1; k <= nz; ++k){
                double x = (i0 + i - 1) * hx;
                double y = (j0 + j - 1) * hy;
                double z = (k0 + k - 1) * hz;
                size_t id = I(i,j,k);
                double val = an.phi(x,y,z);
                h_u_nm1[id] = val; 
                h_u_n[id]   = val;
            }
        }
    }

   
    grid.exchange(h_u_n.data(), timer);

    for(int i = 1; i <= nx; ++i){
        for(int j = 1; j <= ny; ++j){
            for(int k = 1; k <= nz; ++k){
                size_t id   = I(i,j,k);
                size_t id_im = I(i-1,j,k);
                size_t id_ip = I(i+1,j,k);
                size_t id_jm = I(i,j-1,k);
                size_t id_jp = I(i,j+1,k);
                size_t id_km = I(i,j,k-1);
                size_t id_kp = I(i,j,k+1);

                double u0 = h_u_n[id];
                double lap =
                    (h_u_n[id_im] - 2.0*u0 + h_u_n[id_ip]) * cxx +
                    (h_u_n[id_jm] - 2.0*u0 + h_u_n[id_jp]) * cyy +
                    (h_u_n[id_km] - 2.0*u0 + h_u_n[id_kp]) * czz;

                h_u_np1[id] = u0 + 0.5 * lap;
            }
        }
    }
    
    grid.exchange(h_u_np1.data(), timer);

    h_u_nm1.swap(h_u_n);
    h_u_n.swap(h_u_np1);

    size_t total_bytes = h_u_n.size() * sizeof(double);
    cuda_check(cudaMemcpy(d_u_nm1, h_u_nm1.data(), total_bytes, cudaMemcpyHostToDevice), "Init nm1");
    cuda_check(cudaMemcpy(d_u_n, h_u_n.data(), total_bytes, cudaMemcpyHostToDevice), "Init n");
    
    double t1 = MPI_Wtime();
    timer.init_time += (t1 - t0);
}

void SolverCUDA::step(){
    int nx = grid.nx;
    int ny = grid.ny;
    int nz = grid.nz;
    int ny2 = ny + 2;
    int nz2 = nz + 2;

    double t0, t1;

    t0 = MPI_Wtime();
    {
        dim3 block(128, 1, 1);
        dim3 grid_dim((nz + 127)/128, ny, 1);
        pack_x_kernel<<<grid_dim, block>>>(d_u_n, d_send_xm, 1, ny, nz, ny2, nz2);
        pack_x_kernel<<<grid_dim, block>>>(d_u_n, d_send_xp, nx, ny, nz, ny2, nz2);
    }
    {
        dim3 block(128, 1, 1);
        dim3 grid_dim((nz + 127)/128, nx, 1);
        pack_y_kernel<<<grid_dim, block>>>(d_u_n, d_send_ym, 1, nx, nz, ny2, nz2);
        pack_y_kernel<<<grid_dim, block>>>(d_u_n, d_send_yp, ny, nx, nz, ny2, nz2);
    }
    {
        dim3 block(32, 32, 1);
        dim3 grid_dim((ny + 31)/32, (nx+31)/32, 1); 
        if(grid.nbr_zm != MPI_PROC_NULL)
            pack_z_kernel<<<grid_dim, block>>>(d_u_n, d_send_zm, 1, nx, ny, ny2, nz2);
        if(grid.nbr_zp != MPI_PROC_NULL)
            pack_z_kernel<<<grid_dim, block>>>(d_u_n, d_send_zp, nz, nx, ny, ny2, nz2);
    }
    cuda_check(cudaDeviceSynchronize(), "Pack sync");
    t1 = MPI_Wtime();
    timer.comm_pack_time += (t1 - t0);
    t0 = MPI_Wtime();
    size_t sz_x = ny * nz * sizeof(double);
    size_t sz_y = nx * nz * sizeof(double);
    size_t sz_z = nx * ny * sizeof(double);
    
    cudaMemcpy(h_send_xm.data(), d_send_xm, sz_x, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_send_xp.data(), d_send_xp, sz_x, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_send_ym.data(), d_send_ym, sz_y, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_send_yp.data(), d_send_yp, sz_y, cudaMemcpyDeviceToHost);
    if(grid.nbr_zm != MPI_PROC_NULL) cudaMemcpy(h_send_zm.data(), d_send_zm, sz_z, cudaMemcpyDeviceToHost);
    if(grid.nbr_zp != MPI_PROC_NULL) cudaMemcpy(h_send_zp.data(), d_send_zp, sz_z, cudaMemcpyDeviceToHost);
    
    t1 = MPI_Wtime();
    timer.copy_d2h_time += (t1 - t0);

    MPI_Request reqs[12];
    int rcount = 0;
    t0 = MPI_Wtime();
    
    MPI_Isend(h_send_xm.data(), ny*nz, MPI_DOUBLE, grid.nbr_xm, 100, MPI_COMM_WORLD, &reqs[rcount++]);
    MPI_Irecv(h_recv_xm.data(), ny*nz, MPI_DOUBLE, grid.nbr_xm, 101, MPI_COMM_WORLD, &reqs[rcount++]);
    MPI_Isend(h_send_xp.data(), ny*nz, MPI_DOUBLE, grid.nbr_xp, 101, MPI_COMM_WORLD, &reqs[rcount++]);
    MPI_Irecv(h_recv_xp.data(), ny*nz, MPI_DOUBLE, grid.nbr_xp, 100, MPI_COMM_WORLD, &reqs[rcount++]);


    MPI_Isend(h_send_ym.data(), nx*nz, MPI_DOUBLE, grid.nbr_ym, 200, MPI_COMM_WORLD, &reqs[rcount++]);
    MPI_Irecv(h_recv_ym.data(), nx*nz, MPI_DOUBLE, grid.nbr_ym, 201, MPI_COMM_WORLD, &reqs[rcount++]);
    MPI_Isend(h_send_yp.data(), nx*nz, MPI_DOUBLE, grid.nbr_yp, 201, MPI_COMM_WORLD, &reqs[rcount++]);
    MPI_Irecv(h_recv_yp.data(), nx*nz, MPI_DOUBLE, grid.nbr_yp, 200, MPI_COMM_WORLD, &reqs[rcount++]);

    if(grid.nbr_zm != MPI_PROC_NULL){
        MPI_Isend(h_send_zm.data(), nx*ny, MPI_DOUBLE, grid.nbr_zm, 300, MPI_COMM_WORLD, &reqs[rcount++]);
        MPI_Irecv(h_recv_zm.data(), nx*ny, MPI_DOUBLE, grid.nbr_zm, 301, MPI_COMM_WORLD, &reqs[rcount++]);
    }
    if(grid.nbr_zp != MPI_PROC_NULL){
        MPI_Isend(h_send_zp.data(), nx*ny, MPI_DOUBLE, grid.nbr_zp, 301, MPI_COMM_WORLD, &reqs[rcount++]);
        MPI_Irecv(h_recv_zp.data(), nx*ny, MPI_DOUBLE, grid.nbr_zp, 300, MPI_COMM_WORLD, &reqs[rcount++]);
    }

    t1 = MPI_Wtime();
    timer.comm_send_time += (t1 - t0);

    t0 = MPI_Wtime();
    MPI_Waitall(rcount, reqs, MPI_STATUSES_IGNORE);
    t1 = MPI_Wtime();
    timer.comm_wait_time += (t1 - t0);

    t0 = MPI_Wtime();
    cudaMemcpy(d_recv_xm, h_recv_xm.data(), sz_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_recv_xp, h_recv_xp.data(), sz_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_recv_ym, h_recv_ym.data(), sz_y, cudaMemcpyHostToDevice);
    cudaMemcpy(d_recv_yp, h_recv_yp.data(), sz_y, cudaMemcpyHostToDevice);
    if(grid.nbr_zm != MPI_PROC_NULL) cudaMemcpy(d_recv_zm, h_recv_zm.data(), sz_z, cudaMemcpyHostToDevice);
    else cudaMemset(d_recv_zm, 0, sz_z);
    
    if(grid.nbr_zp != MPI_PROC_NULL) cudaMemcpy(d_recv_zp, h_recv_zp.data(), sz_z, cudaMemcpyHostToDevice);
    else cudaMemset(d_recv_zp, 0, sz_z); 

    t1 = MPI_Wtime();
    timer.copy_h2d_time += (t1 - t0);

    t0 = MPI_Wtime();
    {
        dim3 block(128, 1, 1);
        dim3 grid_dim((nz + 127)/128, ny, 1);
        unpack_x_kernel<<<grid_dim, block>>>(d_u_n, d_recv_xm, 0, ny, nz, ny2, nz2);
        unpack_x_kernel<<<grid_dim, block>>>(d_u_n, d_recv_xp, nx+1, ny, nz, ny2, nz2);
    }
    {
        dim3 block(128, 1, 1);
        dim3 grid_dim((nz + 127)/128, nx, 1);
        unpack_y_kernel<<<grid_dim, block>>>(d_u_n, d_recv_ym, 0, nx, nz, ny2, nz2);
        unpack_y_kernel<<<grid_dim, block>>>(d_u_n, d_recv_yp, ny+1, nx, nz, ny2, nz2);
    }
    {
        dim3 block(32, 32, 1);
        dim3 grid_dim((ny + 31)/32, (nx+31)/32, 1);
        unpack_z_kernel<<<grid_dim, block>>>(d_u_n, d_recv_zm, 0, nx, ny, ny2, nz2);
        unpack_z_kernel<<<grid_dim, block>>>(d_u_n, d_recv_zp, nz+1, nx, ny, ny2, nz2);
    }
    cuda_check(cudaDeviceSynchronize(), "Unpack sync");
    t1 = MPI_Wtime();
    timer.comm_unpack_time += (t1 - t0);

    t0 = MPI_Wtime();
    {
        dim3 block(128, 1, 1);
        dim3 grid_opt(
            (nz + 127)/128, 
            ny, 
            nx
        ); 
        wave_step_opt_kernel<<<grid_opt, block>>>(
            d_u_nm1, d_u_n, d_u_np1,
            nx, ny, nz,
            cxx, cyy, czz
        );
    }
    cuda_check(cudaGetLastError(), "wave_step_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "wave_step_kernel sync");
    t1 = MPI_Wtime();
    timer.compute_time += (t1 - t0);

    std::swap(d_u_nm1, d_u_n);
    std::swap(d_u_n, d_u_np1);
}

double SolverCUDA::compute_global_error(double t){
    int nx = grid.nx;
    int ny = grid.ny;
    int nz = grid.nz;

    auto xr = grid.range_x();
    auto yr = grid.range_y();
    auto zr = grid.range_z();
    int i0 = std::get<0>(xr);
    int j0 = std::get<0>(yr);
    int k0 = std::get<0>(zr);

    int threads = 256;
    int blocks = 256; 
    int total_points = nx * ny * nz;
    if (total_points < threads) blocks = 1;

   
    int warps_per_block = threads / 32;
    int total_warps = blocks * warps_per_block;
    
    
    double* d_partials = nullptr;
    cudaMalloc(&d_partials, total_warps * sizeof(double));
    cudaMemset(d_partials, 0, total_warps * sizeof(double));

    error_kernel<<<blocks, threads>>>(
        d_u_n, d_partials,
        nx, ny, nz,
        cfg.hx, cfg.hy, cfg.hz,
        i0, j0, k0,
        t,
        cfg.Lx, cfg.Ly, cfg.Lz,
        cfg.omega
    );
    cuda_check(cudaGetLastError(), "error kernel");

    std::vector<double> h_partials(total_warps);
    cudaMemcpy(h_partials.data(), d_partials, total_warps * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_partials);

    double local_max = 0.0;
    for(double v : h_partials){
        if(v > local_max) local_max = v;
    }

    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_max;
}

void SolverCUDA::run(){
    const int rank = grid.rank;

    double t_total0 = MPI_Wtime();

    init_layers();

    const int steps = static_cast<int>(std::ceil(cfg.T / cfg.dt));
    double t = cfg.dt;

    if(rank == 0){
        std::cout << "MPI+CUDA Optimized. Nx,Ny,Nz="<<cfg.Nx<<","<<cfg.Ny<<","<<cfg.Nz
                  << "  Lx,Ly,Lz="<<cfg.Lx<<","<<cfg.Ly<<","<<cfg.Lz
                  << "  a="<<cfg.a
                  << "  dt="<<cfg.dt
                  << std::endl;
    }

    double err0 = compute_global_error(t);
    if(rank == 0){
        std::cout << "step 1"
                  << "  t=" << t
                  << "  error_max=" << err0
                  << "  step_time=(init)" << std::endl;
    }

    for(int n = 2; n <= steps; ++n){
        double t0 = MPI_Wtime();
        step();
        double t1 = MPI_Wtime();

        double step_local = t1 - t0;
        double step_max = 0.0;
        MPI_Allreduce(&step_local, &step_max, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        t += cfg.dt;
        
        double err = compute_global_error(t);

        if(rank == 0){
            std::cout << "step " << n
                      << "  t=" << t
                      << "  error_max=" << err
                      << "  step_time=" << step_max << " s\n";
        }
    }

    double t_total1 = MPI_Wtime();
    timer.total_time = t_total1 - t_total0;
    
    if(rank == 0){
        std::cout << "--------------------------------------------------\n";
        std::cout << "TIMING REPORT (Rank 0)\n";
        std::cout << "Total Time:       " << timer.total_time << " s\n";
        std::cout << "Initialization:   " << timer.init_time << " s\n";
        std::cout << "Compute (Kernel): " << timer.compute_time << " s\n";
        std::cout << "Copy H2D:         " << timer.copy_h2d_time << " s\n";
        std::cout << "Copy D2H:         " << timer.copy_d2h_time << " s\n";
        std::cout << "Communication:\n";
        std::cout << "  Pack:           " << timer.comm_pack_time << " s\n";
        std::cout << "  Send (Overhead):" << timer.comm_send_time << " s\n";
        std::cout << "  Wait (Block):   " << timer.comm_wait_time << " s\n";
        std::cout << "  Unpack:         " << timer.comm_unpack_time << " s\n";
        std::cout << "  Total Comm:     " << timer.comm_total_time << " s\n";
        std::cout << "--------------------------------------------------\n";
    }
}

