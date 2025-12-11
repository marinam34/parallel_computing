#!/bin/bash

NX=128
NY=128
NZ=130
LX=1.0
LY=1.0
LZ=1.0
T_FINAL=2.0
CFL=0.95

PROCS_LIST=(1 2 4 8 16 32)

EXECUTABLE="$(pwd)/build/task_mpi"

mkdir -p results tmp build

for np in "${PROCS_LIST[@]}"; do
  JOB_SCRIPT="tmp/job_np_${np}.lsf"

  cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
#BSUB -J mpi_job_${np}
#BSUB -n ${np}
#BSUB -W 00:15
#BSUB -o results/grid_${NX}_${NY}_${NZ}_proc_${np}.%J.out
#BSUB -e results/grid_${NX}_${NY}_${NZ}_proc_${np}.%J.err
#BSUB -R "span[ptile=16]"

module purge
module load openmpi-2.1.3 || { module load openmpi/2.1.2/2018; export OMPI_CXX=g++; }

mkdir -p build
mpicxx -O3 -std=c++11 -I./include -I./src \
  src/main_mpi.cpp src/config.cpp src/grid_mpi.cpp src/solver_mpi.cpp src/analytic.cpp \
  -o "${EXECUTABLE}"

mpirun -np ${np} "${EXECUTABLE}" \
  --Lx ${LX} --Ly ${LY} --Lz ${LZ} \
  --Nx ${NX} --Ny ${NY} --Nz ${NZ} \
  --T ${T_FINAL} --cfl ${CFL}
EOF

  bsub < "${JOB_SCRIPT}"
done

# rm -r tmp   
