#!/bin/bash

LX=1.0
LY=1.0
LZ=1.0
NX=128
NY=128
NZ=130     
T_FINAL=2.0
CFL=0.95

NP_LIST=(1 2 4 8 16 32)
OMP_LIST=(8 8 4 2 2 1)

EXECUTABLE="$(pwd)/build/task_hybrid"

mkdir -p results tmp build

for idx in "${!NP_LIST[@]}"; do
  np=${NP_LIST[$idx]}
  omp=${OMP_LIST[$idx]}
  cores=$((np*omp))
  JOB_SCRIPT="tmp/job_np${np}_omp${omp}.lsf"

  cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
#BSUB -J hybr_np${np}_omp${omp}
#BSUB -n ${cores}
#BSUB -W 00:20
#BSUB -o results/hybrid_${NX}_${NY}_${NZ}_np${np}_t${omp}.%J.out
#BSUB -e results/hybrid_${NX}_${NY}_${NZ}_np${np}_t${omp}.%J.err
#BSUB -R "span[ptile=16]"

module purge
module load openmpi-2.1.3 || { module load openmpi/2.1.2/2018; export OMPI_CXX=g++; }

mkdir -p build
mpicxx -O3 -std=c++11 -fopenmp -I./include -I./src \
  src/main_hybrid.cpp src/config.cpp src/grid_mpi.cpp src/solver_hybrid.cpp src/analytic.cpp \
  -o "${EXECUTABLE}"

export OMP_NUM_THREADS=${omp}
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

mpirun -np ${np} --bind-to core --map-by slot \
  "${EXECUTABLE}" \
  --Lx ${LX} --Ly ${LY} --Lz ${LZ} \
  --Nx ${NX} --Ny ${NY} --Nz ${NZ} \
  --T ${T_FINAL} --cfl ${CFL}
EOF

  bsub < "${JOB_SCRIPT}"
done

# rm -r tmp
