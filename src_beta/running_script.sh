#!/bin/bash
#mpirun -np 2 -hostfile /home/yidawang/code/fcma-toolbox/src/host_special -perhost 1 /home/yidawang/code/fcma-toolbox/src/pni_fcma -d /home/yidawang/data/attloc/data/ -m .nii.gz -t /home/yidawang/data/attloc/result/all -l 18 -s 200 -k 0 -h 0 -n 30 -e /home/yidawang/data/attloc/blockfiles/ -x /home/yidawang/data/attloc/masks/top10.nii.gz -y /home/yidawang/data/attloc/masks/wholebrain.nii.gz
for i in $(seq 1 18)
do
  let lid=(i-1)*12
  mpirun -np 49 -hostfile /home/yidawang/code/fcma-toolbox/src/host -perhost 1 -genv I_MPI_FABRICS shm:tcp /home/yidawang/code/fcma-toolbox/src/pni_fcma -d /home/yidawang/data/face_scene/raw/ -m .nii.gz -t /home/yidawang/data/face_scene_results/raw/original12/sub$i -l $lid -s 200 -k 0 -h 12 -n 17 -b /home/yidawang/code/fcma-toolbox/src/blocks.txt -x /home/yidawang/data/face_scene/masks/masks.nii.gz -y /home/yidawang/data/face_scene/masks/masks.nii.gz
done
#/home/yidawang/code/fcma-toolbox/src/pni_fcma -d /home/yidawang/data/attloc/data/ -m nii.gz -l 0 -c 1 -k 7 -h 18 -n 29 -e /home/yidawang/data/attloc/blockfiles/ -x /home/yidawang/data/attloc/masks/clustp5top1000_10vox.nii.gz -y /home/yidawang/data/attloc/masks/wholebrain.nii.gz -q 1 #-t /home/yidawang/data/attloc/test/wholebrain/topvox/no0412121_topvoxels_list.txt
#/home/yidawang/code/fcma-toolbox/src/pni_fcma -d /home/yidawang/data/face_scene/raw/ -m nii.gz -l 0 -c 1 -k 0 -h 12 -n 17 -b /home/yidawang/code/fcma-toolbox/src/blocks.txt -x /home/yidawang/data/face_scene/masks/mask.nii.gz -y /home/yidawang/data/face_scene/masks/mask.nii.gz -q 1 -t /home/yidawang/data/face_scene_results/raw/sub1_list.txt
