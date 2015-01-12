#!/bin/bash
#SBATCH -n 1 -N 1
export MIC_PPN=1
export MIC_OMP_NUM_THREADS=240
export OMP_NUM_THREADS=32
metarun -2 ./run_mic.sh -c ./run_cpu.sh
