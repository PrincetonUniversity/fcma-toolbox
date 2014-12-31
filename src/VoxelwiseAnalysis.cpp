/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "VoxelwiseAnalysis.h"
#include "Preprocessing.h"
#include "MatComputation.h"
#include "common.h"

/********************************
compute coorelation between "step" number of voxels from the starting row of the first matrix array and the second matrix array, across all trials
one voxel struct contains the correlation between a voxel of matrices1 and all voxels of matrice2, across all trials
input: the trial struct array, number of trials, the starting row (since distributing to multi-nodes), step, the raw matrix struct arrays
output: the voxel struct array
*********************************/
Voxel** ComputeAllVoxelsAnalysisData(Trial* trials, int nTrials, int sr, int step, RawMatrix** matrices1, RawMatrix** matrices2)
{
  int i;
  Voxel** voxels = new Voxel*[step];
  for (i=0; i<step; i++)
  {
    voxels[i] = new Voxel(sr+i, nTrials, matrices2[0]->row);  // assume the number of voxels (row) is the same accross blocks
    voxels[i]->corr_vecs = new float[nTrials*matrices2[0]->row];
  }
  #pragma omp parallel for
  for (i=0; i<nTrials; i++)
  {
    int sc = trials[i].sc;
    int ec = trials[i].ec;
    int sid = trials[i].sid;
    int row1 = matrices1[sid]->row;
    int col1 = matrices1[sid]->col;
    int row2 = matrices2[sid]->row;
    int col2 = matrices2[sid]->col;
    float* buf1 = (float*)_mm_malloc(sizeof(float)*row1*(ec-sc+1), 64);
    int ml = getBuf(sc, ec, row1, col1, matrices1[sid]->matrix, buf1);
    float* buf2 = (float*)_mm_malloc(sizeof(float)*row2*(ec-sc+1), 64);
    getBuf(sc, ec, row2, col2, matrices2[sid]->matrix, buf2);
    for (int j=0; j<step; j++)
    {
      vectorMatMultiply(buf2, sizeof(float)*row2*ml, buf1+(sr+j)*ml, sizeof(float)*ml, (voxels[j]->corr_vecs)+i*row2, sizeof(float)*row2);
    }
    _mm_free(buf1);
    _mm_free(buf2);
  }
  return voxels;
}

/******************************
compute a voxel of id "vid", across all trials
input: a trial struct, voxel id, number of trials, normalized data, number of voxels (row), number of TRs (col)
output: the voxel struct
*******************************/
Voxel* ComputeOneVoxelAnalysisData(Trial* trials, int vid, int nTrials, float** data_buf1, float** data_buf2, int row, int col)
{
  Voxel* voxel = new Voxel(vid, nTrials, row);
  voxel->corr_vecs = new float[nTrials*row];
  int i;
  for (i=0; i<nTrials; i++)
  {
    int sc = trials[i].sc;
    int ec = trials[i].ec;
    int sid = trials[i].sid;
    float* buf1 = data_buf1[i];
    float* buf2 = data_buf2[i];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, row, col, 1.0, buf1+vid*col, col, buf2, col, 0.0, voxel->corr_vecs+i*row, row);
    // c=a*b', a: n1*n2, b: n3*n2, c: n1*n3
    //matmul(buf1+vid*col, buf2, voxel->corr_vecs+i*row, 1, col, row);
  }
  return voxel;
}

/****************************************
Fisher transform the correlation values (coefficients) then z-scored them across within-subject blocks for all voxels' correlation data
input: the voxel array, the length of this array (the number of voxels, "step"), the number of subjects
output: update values to the voxels' correlation data
*****************************************/
void PreprocessAllVoxelsAnalysisData(Voxel** voxels, int step, int nSubs)
{    
  int i;
  #pragma omp parallel for private(i)
  for (i=0; i<step; i++)
  {
    PreprocessOneVoxelsAnalysisData(voxels[i], nSubs);
  }
  return;
}

/****************************************
Fisher transform the correlation values (coefficients) then z-scored them across within-subject blocks for one voxel's correlation data
input: the voxel, the number of subjects
output: update values to the voxel's correlation data
*****************************************/
void PreprocessOneVoxelsAnalysisData(Voxel* voxel, int nSubs)
{
  int i, j, k;
  int nTrials = voxel->nTrials;
  int row = voxel->nVoxels;
  int nPerSub=nTrials/nSubs;
  ALIGNED(64) float buf[nPerSub];
  for (i=0; i<nSubs; i++)
  {
    for (j=0; j<row; j++)
    {
      #pragma simd  // simd and prefetch and simd+preftech got similar performance
      for (k=0; k<nPerSub; k++)
      {
        buf[k] = fisherTransformation(voxel->corr_vecs[i*nPerSub*row+j+k*row]);
      }
      if (nPerSub>1)  // nPerSub==1 results in 0
        z_score(buf, nPerSub);
      #pragma vector aligned nontemporal  // stream storing helps a little
      for (k=0; k<nPerSub; k++)
      {
        voxel->corr_vecs[i*nPerSub*row+j+k*row] = buf[k];
      }
    }
  }
  return;
}

/****************************************
Perform vector matrix multiply for fine-grained threads
input: matrix data, matrix data size, vector data, vector data size, output data location, output data size (all sizes are in bytes)
output: update values to the output data
*****************************************/
void vectorMatMultiply(float* mat, int mat_size, float* vec, int vec_size, float* output, int output_size)
{
  int col = vec_size/sizeof(float);
  int row = mat_size/sizeof(float)/col;
  for (int i=0; i<row; i++)
  {
    float sum=0.0;  // use an additional variable for reduction in the following vectorization
    #pragma loop count(12)
    for (int j=0; j<col; j++)
    {
      sum += mat[i*col+j]*vec[j];
    }
    output[i]=sum;
  }
  return;
}
