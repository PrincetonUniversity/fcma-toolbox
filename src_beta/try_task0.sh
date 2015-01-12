#!/bin/bash
for i in $(seq $1 $2)
do
  let lid=(i-1)*12
  mpiexec.hydra -np 18 -machinefile host -perhost 1 -genv I_MPI_FABRICS shm:tcp /home/yidawang/code/fcma-toolbox/src_mpi/pni_fcma -d /home/yidawang/data/face_scene/raw/ -m .nii.gz -t test_sub$i -l $lid -s 60 -k 0 -h 12 -n 17 -b /home/yidawang/code/fcma-toolbox/src_mpi/blocks.txt -x /home/yidawang/data/face_scene/masks/masks.nii.gz -y /home/yidawang/data/face_scene/masks/masks.nii.gz
done
