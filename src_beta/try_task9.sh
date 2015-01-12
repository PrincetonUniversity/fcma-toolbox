#!/bin/bash
for i in $(seq $1 $2)
do
  let lid=(i-1)*12
  mpirun -np 2 -hostfile /home/yidawang/code/fcma-toolbox/src_mpi/host -perhost 1 -genv I_MPI_FABRICS shm:tcp /home/yidawang/code/fcma-toolbox/src_mpi/pni_fcma -d /home/yidawang/data/face_scene/raw/ -m .nii.gz -t try_sub$i -l $lid -s 720 -k 9 -h 12 -n 17 -b /home/yidawang/code/fcma-toolbox/src_mpi/blocks.txt -x /home/yidawang/data/face_scene/masks/masks.nii.gz
done
