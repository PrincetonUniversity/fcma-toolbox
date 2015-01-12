#!/bin/bash
if [ $# -ne 4 ]; then
  echo "input arguments error"
  echo "first arg: #nodes"
  echo "second arg: hostfile"
  echo "third arg: #voxels processed at once"
  echo "fourth arg: task number, 0-voxel selection, 9-logistic regression"
  exit 0
fi
if [ $4 -ne 0 ] && [ $4 -ne 9 ]; then
  echo "fourth arg: task number, 0-voxel selection, 9-logistic regression"
  exit 0
fi

export I_MPI_MIC=enable
export I_MPI_MIC_POSTFIX='.MIC'
export LD_LIBRARY_PATH=/opt/intel/composer_xe/compiler/lib/mic/:/opt/intel/mkl/lib/mic/:$LD_LIBRARY_PATH
#export PATH=/opt/intel/impi/4.1.1.036/mic/bin/:$PATH

for i in $(seq 1 30)
do
  let lid=(i-1)*18
  mpiexec.hydra \
    -np $1 -machinefile $2 -perhost 1 \
    -genv I_MPI_FABRICS shm:tcp \
    ./pni_fcma -d /home/yidawang/data/attloc/data/ \
    -m .nii.gz -t /home/yidawang/data/attloc/result/marginal_screening/sub$i -l $lid -s $3 -k $4 -h 18 -n 29 \
    -e /home/yidawang/data/attloc/blockfiles/ \
    -x /home/yidawang/data/attloc/masks/wholebrain.nii.gz \
    -y /home/yidawang/data/attloc/masks/wholebrain.nii.gz
done
