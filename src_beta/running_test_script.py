import os, sys

for i in range(1, 19):
  lid = (i-1)*12
  cmd = './pni_fcma -d /home/yidawang/data/face_scene/raw/ -m .nii.gz -t ~/data/face_scene_results/raw/sub' + str(i) + '_list.txt -l ' + str(lid) + ' -c 1 -q 1 -k 0 -h 12 -n 17 -b blocks.txt -x /home/yidawang/data/face_scene/masks/mask.nii.gz -y /home/yidawang/data/face_scene/masks/mask.nii.gz >> /home/yidawang/data/face_scene_results/act.log'
  print cmd
  os.system(cmd)
