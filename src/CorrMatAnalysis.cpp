/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "CorrMatAnalysis.h"
#include "Scheduler.h"
#include "common.h"

/****************************************
get the average correlation coefficients of correlation vectors accross blocks for every voxel
input: the node id, the correlation matrix array (normally all belong to one subject), the number of blocks
output: a list of voxels' scores in terms of average correlation coefficients
*****************************************/
VoxelScore* GetCorrVecSum(int me, CorrMatrix** c_matrices, int nTrials)  //scores for a c_matrix array
{
  if (me==0)  //sanity check
  {
    cerr<<"the master node isn't supposed to do classification jobs"<<endl;
    exit(1);
  }
  int rowBase = c_matrices[0]->sr;  // assume all elements in c_matrices array have the same starting row
  int row = c_matrices[0]->nVoxels; // assume all elements in c_matrices array have the same #voxels
  int length = row * c_matrices[0]->step; // assume all elements in c_matrices array have the same step, get the number of entries of a coorelation matrix, notice the row here!!
  VoxelScore* scores = new VoxelScore[c_matrices[0]->step];  // get step voxels' scores here
  #pragma omp parallel for
  for (int i=0; i<length; i+=row)
  {
    int count = i / row;  // write count in this way for parallelization
    (scores+count)->vid = rowBase+i/row;
    (scores+count)->score = AllTrialsCorrVecSum(nTrials, i, c_matrices, row); // compute the sum for one voxel
  }
  return scores;
}

/*****************************************
get the average correlation coefficients of correlation vectors across blocks for one voxel
input: the number of blocks, the voxel id, the correlation matrix array, the number of entries in a partial correlation matrix
output: the average correlation coefficient of this voxel
******************************************/
float AllTrialsCorrVecSum(int nTrials, int startIndex, CorrMatrix** c_matrices, int length)
{
  int i;
  float result = 0.0;
  for (i=0; i<nTrials; i++)
  {
    result += GetVectorSum(&(c_matrices[i]->matrix[startIndex]), length);
  }
  result /= nTrials;
  result /= length;
  return result;
}

/*****************************************
get the summation of all elements of a vector
input: a vector and its length
output: the summation of this vector
******************************************/
float GetVectorSum(float* v, int length)
{
  int i;
  float result = 0.0;
  for (i=0; i<length; i++)
  {
    if (!isnan(v[i]))
    {
      result += v[i];
    }
  }
  return result;
}
