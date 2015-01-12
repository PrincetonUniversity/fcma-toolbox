#!/bin/bash
export I_MPI_MIC=enable
export I_MPI_MIC_POSTFIX='.MIC'
export LD_LIBRARY_PATH=/opt/intel/composer_xe/compiler/lib/mic/:/opt/intel/mkl/lib/mic/:$LD_LIBRARY_PATH
#export PATH=/opt/intel/impi/4.1.1.036/mic/bin/:$PATH
for i in $(seq $1 $2)
do
  let lid=(i-1)*12
  #mpirun -np 14 -hosts n01,n02-mic1,n02-mic0,n01-mic1,n01-mic0,n00-mic0,n00-mic1,n03-mic1,n05-mic0,n05-mic1,n04-mic0,n04-mic1,n30-mic0,n30-mic1 -perhost 1 -genv I_MPI_FABRICS shm:tcp /mpich-test/pni_fcma -d /home/yidawang/data/face_scene/raw/ -m .nii.gz -t test_sub$i -l $lid -s 40 -k 0 -h 12 -n 17 -b /home/yidawang/code/fcma-toolbox/src_mpi/blocks.txt -x /home/yidawang/data/face_scene/masks/masks.nii.gz -y /home/yidawang/data/face_scene/masks/masks.nii.gz
  mpiexec.hydra \
    -np 2 -machinefile mic-host -perhost 1 \
    -genv I_MPI_FABRICS shm:tcp \
    ./pni_fcma -d /home/yidawang/data/face_scene/raw/ \
    -m .nii.gz -t try_sub$i -l $lid -s 720 -k 9 -h 12 -n 17 \
    -b /home/yidawang/code/fcma-toolbox/src_mpi/blocks.txt \
    -x /home/yidawang/data/face_scene/masks/masks.nii.gz 
done
