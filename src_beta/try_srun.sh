#!/bin/bash
if [ $# -ne 3 ]; then
  echo "input arguments error"
  echo "first arg: #nodes"
  echo "second arg: #voxels processed at once"
  echo "third arg: task number, 0-voxel selection, 3-correlation sum, 9-logistic regression"
  exit 0
fi
if [ $3 -ne 0 ] && [ $3 -ne 3 ] && [ $3 -ne 9 ]; then
  echo "fourth arg: task number, 0-voxel selection, 3-correlation sum, 9-logistic regression"
  exit 0
fi
#for i in $(seq 1 1)
#do
  module load intel
  let lid=0
  srun -n $1 -c 32 -p fcma ./pni_fcma -d /home/yidawang/data/face_scene/raw/ \
    -m .nii.gz -t try_sub1 -l $lid -s $2 -k $3 -h 12 -n 17 \
    -b /home/yidawang/code/fcma-toolbox/src\_th/blocks.txt \
    -x /home/yidawang/data/face_scene/masks/masks.nii.gz \
    -y /home/yidawang/data/face_scene/masks/masks.nii.gz
