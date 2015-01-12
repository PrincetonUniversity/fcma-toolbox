#!/bin/bash
./pni_fcma -d /home/yidawang/data/face_scene/raw/ \
    -m .nii.gz -t try_sub1 -l 0 -s 60 -k 0 -h 12 -n 17\
    -b /home/yidawang/code/fcma-toolbox/src/blocks.txt \
    -x /home/yidawang/data/face_scene/masks/masks.nii.gz \
    -y /home/yidawang/data/face_scene/masks/masks.nii.gz
