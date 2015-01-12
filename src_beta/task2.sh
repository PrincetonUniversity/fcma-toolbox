#!/bin/bash
for i in $(seq $1 $2)
do
  let lid=(i-1)*12
  /home/yidawang/code/fcma-toolbox/src/pni_fcma -d /home/yidawang/data/face_scene/residual/ -m .nii.gz -t /home/yidawang/data/face_scene_results/residual/original12_acti/sub$i -l $lid -s 200 -k 2 -h 12 -n 17 -b /home/yidawang/code/fcma-toolbox/src/blocks_res_12.txt -x /home/yidawang/data/face_scene/masks/masks.nii.gz
done
