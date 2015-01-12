#!/bin/bash
for i in $(seq $1 $2)
do
  let lid=(i-1)*18
  mpirun -np 49 -hostfile /home/yidawang/code/fcma-toolbox/src_mpi/host -perhost 1 -genv I_MPI_FABRICS shm:tcp /home/yidawang/code/fcma-toolbox/src_mpi/pni_fcma -d /home/yidawang/data/attloc/data/ -m .nii.gz -t try_sub$i -l $lid -s 240 -k $3 -h 18 -n 29 -e /home/yidawang/data/attloc/blockfiles/ -x /home/yidawang/data/attloc/masks/wholebrain.nii.gz -y /home/yidawang/data/attloc/masks/wholebrain.nii.gz 
done
