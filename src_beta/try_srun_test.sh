#!/bin/bash
#for i in $(seq 1 1)
#do
  module load intel
  let lid=($1-1)*12
  srun -n 1 -c 32 -p fcma ./pni_fcma -d /home/yidawang/data/face_scene/raw/ \
    -m .nii.gz -t /home/yidawang/data/face_scene/results/raw/first19/sub$1\_list.txt -l $lid -k 0 -h 12 -n 17 -c 1 -q 1\
    -b /home/yidawang/code/fcma-toolbox/src/blocks.txt \
    -x /home/yidawang/data/face_scene/masks/masks.nii.gz \
    -y /home/yidawang/data/face_scene/masks/masks.nii.gz #> /home/yidawang/data/face_scene/results/residual/first19/sub$1\_result.txt
