#!/bin/bash
NX=128; NY=128; NZ=128
LX=1.0; LY=1.0; LZ=1.0
T_FINAL=2.0; CFL=0.50
EXECUTABLE="build/solver_omp"

THREADS_LIST=(1 2 4 8 16 32)
FALLBACK_CORES_LIST=(1 1 1 1 3 5)   # твоя рабочая схема

mkdir -p results tmp

submit_job () {
  local threads="$1" cores="$2" tag="$3"
  local job="tmp/job_${threads}_${tag}.lsf"
  {
    echo "#!/bin/bash"
    echo "#BSUB -J omp_${threads}_${tag}"
    echo "#BSUB -n ${cores}"
    echo "#BSUB -W 00:20"
    echo "#BSUB -R \"span[hosts=1]\""
    echo "#BSUB -o results/grid_${NX}_L_${LX}_thr_${threads}_${tag}.%J.out"
    echo "#BSUB -e results/grid_${NX}_L_${LX}_thr_${threads}_${tag}.%J.err"
    echo "export OMP_NUM_THREADS=${threads}"
    echo "export OMP_DYNAMIC=false"
    echo "export OMP_SCHEDULE=static"
    echo "export OMP_PROC_BIND=close"
    echo "export OMP_PLACES=cores"
    echo "${EXECUTABLE} --Lx ${LX} --Ly ${LY} --Lz ${LZ} --Nx ${NX} --Ny ${NY} --Nz ${NZ} --T ${T_FINAL} --cfl ${CFL}"
  } > "${job}"
  jid=$(bsub < "${job}" 2>&1 | awk '/Job <[0-9]+>/ {gsub(/[<>]/,"",$2); print $2}')
  if [ -n "$jid" ]; then
    echo "submitted ${threads}t ${tag}: job ${jid}"
    return 0
  else
    echo "submit failed ${threads}t ${tag}"
    return 1
  fi
}

for i in "${!THREADS_LIST[@]}"; do
  threads=${THREADS_LIST[i]}

  sed -i '' -e '1s/^/#/' /dev/null 2>/dev/null || true  # noop для совместимости mac sed
  job_pref="tmp/job_${threads}_pref.lsf"
  {
    echo "#!/bin/bash"
    echo "#BSUB -J omp_${threads}_pref"
    echo "#BSUB -n ${threads}"
    echo "#BSUB -x"
    echo "#BSUB -W 00:20"
    echo "#BSUB -R \"span[hosts=1]\""
    echo "#BSUB -o results/grid_${NX}_L_${LX}_thr_${threads}_pref.%J.out"
    echo "#BSUB -e results/grid_${NX}_L_${LX}_thr_${threads}_pref.%J.err"
    echo "export OMP_NUM_THREADS=${threads}"
    echo "export OMP_DYNAMIC=false"
    echo "export OMP_SCHEDULE=static"
    echo "export OMP_PROC_BIND=close"
    echo "export OMP_PLACES=cores"
    echo "${EXECUTABLE} --Lx ${LX} --Ly ${LY} --Lz ${LZ} --Nx ${NX} --Ny ${NY} --Nz ${NZ} --T ${T_FINAL} --cfl ${CFL}"
  } > "${job_pref}"
  jid_pref=$(bsub < "${job_pref}" 2>&1 | awk '/Job <[0-9]+>/ {gsub(/[<>]/,"",$2); print $2}')
  if [ -n "$jid_pref" ]; then
    echo "submitted ${threads}t pref: job ${jid_pref}"
    continue
  fi

  # 2) fallback: ядра — из вашей CORES_LIST, нити — как запрошено
  cores=${FALLBACK_CORES_LIST[i]}
  job_fb="tmp/job_${threads}_fb.lsf"
  {
    echo "#!/bin/bash"
    echo "#BSUB -J omp_${threads}_fb"
    echo "#BSUB -n ${cores}"
    echo "#BSUB -W 00:20"
    echo "#BSUB -R \"span[hosts=1]\""
    echo "#BSUB -o results/grid_${NX}_L_${LX}_thr_${threads}_fb.%J.out"
    echo "#BSUB -e results/grid_${NX}_L_${LX}_thr_${threads}_fb.%J.err"
    echo "export OMP_NUM_THREADS=${threads}"
    echo "export OMP_DYNAMIC=false"
    echo "export OMP_SCHEDULE=static"
    echo "export OMP_PROC_BIND=spread"
    echo "export OMP_PLACES=threads"
    echo "${EXECUTABLE} --Lx ${LX} --Ly ${LY} --Lz ${LZ} --Nx ${NX} --Ny ${NY} --Nz ${NZ} --T ${T_FINAL} --cfl ${CFL}"
  } > "${job_fb}"
  jid_fb=$(bsub < "${job_fb}" 2>&1 | awk '/Job <[0-9]+>/ {gsub(/[<>]/,"",$2); print $2}')
  if [ -n "$jid_fb" ]; then
    echo "submitted ${threads}t fallback: job ${jid_fb}"
  else
    echo "submit failed ${threads}t fallback"
  fi
done

rm -rf tmp

