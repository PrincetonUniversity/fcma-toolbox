#!/bin/bash
for i in $(seq $1 $2)
do
  let lid=(i-1)*12
  mpirun -np 49 -hostfile /home/yidawang/code/fcma-toolbox/src/host -perhost 1 -genv I_MPI_FABRICS shm:tcp /home/yidawang/code/fcma-toolbox/src/pni_fcma -d /home/yidawang/data/face_scene/raw/ -m .nii.gz -t /home/yidawang/data/face_scene_results/raw/original12/sub$i -l $lid -s 200 -k 0 -h 12 -n 17 -b /home/yidawang/code/fcma-toolbox/src/blocks.txt -x /home/yidawang/data/face_scene/masks/masks.nii.gz -y /home/yidawang/data/face_scene/masks/masks.nii.gz
done
