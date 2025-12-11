#!/bin/bash

NX=512
NY=512
NZ=512
LX=1.0
LY=1.0
LZ=1.0
T_FINAL=0.065
CFL=0.95

PROCS_LIST=(2 20)

EXECUTABLE="$(pwd)/build/task_cuda"

mkdir -p results tmp build

for np in "${PROCS_LIST[@]}"; do
  JOB_SCRIPT="tmp/job_np_${np}.lsf"

  cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
#BSUB -J cuda_job_${np}
#BSUB -n ${np}
#BSUB -gpu "num=1"
#BSUB -W 00:15
#BSUB -o results/grid_${NX}_${NY}_${NZ}_cuda_proc_${np}.%J.out
#BSUB -e results/grid_${NX}_${NY}_${NZ}_cuda_proc_${np}.%J.err
#BSUB -R "span[ptile=20]"

module purge
module load cuda
module load openmpi-2.1.3 || { module load openmpi/2.1.2/2018; export OMPI_CXX=g++; }

ARCH=sm_60 HOST_COMP=mpicc make

export CUDA_VISIBLE_DEVICES=0

mpirun -np ${np} "${EXECUTABLE}" \
  --Lx ${LX} --Ly ${LY} --Lz ${LZ} \
  --Nx ${NX} --Ny ${NY} --Nz ${NZ} \
  --T ${T_FINAL} --cfl ${CFL}
EOF

  bsub < "${JOB_SCRIPT}"
done

rm -r tmp

