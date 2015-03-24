/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "VoxelwiseAnalysis.h"
#include "Preprocessing.h"
#include "MatComputation.h"
#include "CustomizedMatrixMultiply.h"
#include "common.h"
#include "SVMClassification.h"

/********************************
compute coorelation between "step" number of voxels from the starting row of the first matrix array and the second matrix array, across all trials
one voxel struct contains the correlation between a voxel of matrices1 and all voxels of matrice2, across all trials
input: the trial struct array, number of trials, number of subjects, number of training trials, the starting row (since distributing to multi-nodes), step, the raw matrix struct arrays
output: the voxel struct array
*********************************/
Voxel* ComputeAllVoxelsAnalysisData(Voxel* voxels, Trial* trials, int nTrials, int nSubs, int nTrainings, int sr, int step, RawMatrix** matrices1, RawMatrix** matrices2, float* bufs1, float* bufs2)
{
  int i;
  //Voxel** voxels = new Voxel*[step];
  for (i=0; i<step; i++)
  {
    voxels->vid[i]=sr+i;
  }
//float t;
//struct timeval start, end;
//gettimeofday(&start, 0);
#if 0
  #pragma omp parallel for
  for (i=0; i<nTrials; i++)
  {
    int cur_col = trials[i].sc; // cur_col=within_subject_i*ml
    int ml = trials[i].ec;
    int sid = trials[i].sid;
    int row1 = matrices1[sid]->row;
    int col1 = matrices1[sid]->col;
    int row2 = matrices2[sid]->row;
    int col2 = matrices2[sid]->col;
    //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, step, row2, ml, 1.0, bufs1[i]+sr*ml, ml, bufs2[i], ml, 0.0, voxels->corr_vecs+i*row2, row2*nTrials);
    //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, step, row2, ml, 1.0, matrices1[sid]->matrix+cur_col*row1+sr*ml, ml, matrices2[sid]->matrix+cur_col*row2, ml, 0.0, voxels->corr_vecs+i*row2, row2*nTrials);
    sgemmTranspose(/*matrices1[sid]->matrix+cur_col*row1+sr*ml*/bufs1+i*row1*ml+sr*ml, /*matrices2[sid]->matrix+cur_col*row2*/bufs2+i*row2*ml, step, row2, ml, voxels->corr_vecs+i*row2, row2*nTrials);
  }
#endif
  int row=matrices1[0]->row;    // assuming all matrices have the same size
  int ml=trials[0].ec;    // assuming all blocks have the same size
  int nPerSubj=nTrials/nSubs;    // assuming all subjects have the same number of blocks
  int m_max = (step/BLK2)*BLK2;
#ifdef __MIC__
  widelock_t Clocks[BLK2];
  #pragma omp parallel for
  for (int i=0; i<BLK2; i++)
  {
    omp_init_lock(&(Clocks[i].lock));
  }
#endif
  for (int m=0; m<step; m+=BLK2)
  {
    if (m<m_max)
    {
      sgemmTransposeMerge(bufs1+(sr+m)*ml, bufs2, BLK2, row, ml, nPerSubj, nSubs, voxels->corr_output, row*nTrials, trials);
#ifdef __MIC__
      int np = BLK2;
      int threads_per_prob = MICTH/np;
      #pragma omp parallel for
      for(int p = 0 ; p < np*threads_per_prob ; p++)
      {
        int prob_ID = p / threads_per_prob;
        int thread_ID = p % threads_per_prob;
        int k_per_thread = ((row + threads_per_prob-1) / threads_per_prob);
        int start_k = thread_ID * k_per_thread;
        int end_k = (thread_ID+1) * k_per_thread;
        if(end_k > row) end_k = row;
        int k_range = end_k - start_k;
        custom_ssyrk((const int)nTrainings, 
           (const int)k_range, voxels->corr_output+prob_ID*row*nTrials + start_k,
           (const int)row, voxels->corr_vecs+(prob_ID+m)*nTrainings*nTrainings, Clocks[prob_ID], (const int)nTrainings);
      }
#else
      #pragma omp parallel for
      for (int mm=m; mm<m+BLK2; mm++)
      {
        //custom_ssyrk_old((const int)nTrainings, (const int)row, voxels->corr_output+(mm-m)*row*nTrials, (const int)row, voxels->corr_vecs+mm*nTrainings*nTrainings, (const int)nTrainings);
        cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, nTrainings, row, 1.0, voxels->corr_output+(mm-m)*row*nTrials, row, 0.0, voxels->corr_vecs+mm*nTrainings*nTrainings, nTrainings);
      }
#endif
    }
    else
    {
      sgemmTransposeMerge(bufs1+(sr+m)*ml, bufs2, step-m_max, row, ml, nPerSubj, nSubs, voxels->corr_output, row*nTrials, trials);
#ifdef __MIC__
      int np = step-m_max;
      int threads_per_prob = MICTH/np;
      #pragma omp parallel for
      for(int p = 0 ; p < np*threads_per_prob ; p++)
      {
        int prob_ID = p / threads_per_prob;
        int thread_ID = p % threads_per_prob;
        int k_per_thread = ((row + threads_per_prob-1) / threads_per_prob);
        int start_k = thread_ID * k_per_thread;
        int end_k = (thread_ID+1) * k_per_thread;
        if(end_k > row) end_k = row;
        int k_range = end_k - start_k;
        custom_ssyrk((const int)nTrials, 
           (const int)k_range, voxels->corr_output+prob_ID*row*nTrainings + start_k,
           (const int)row, voxels->corr_vecs+(prob_ID+m)*nTrainings*nTrainings, Clocks[prob_ID], (const int)nTrainings);
      }
#else
      #pragma omp parallel for
      for (int mm=m; mm<step; mm++)
      {
        //custom_ssyrk_old((const int)nTrainings, (const int)row, voxels->corr_output+(mm-m)*row*nTrials, (const int)row, voxels->corr_vecs+mm*nTrainings*nTrainings, (const int)nTrainings);
        cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, nTrainings, row, 1.0, voxels->corr_output+(mm-m)*row*nTrials, row, 0.0, voxels->corr_vecs+mm*nTrainings*nTrainings, nTrainings);
      }
#endif
    } // else m
  } //for m
  return voxels;
}

/****************************************
Fisher transform the correlation values (coefficients) then z-scored them across within-subject blocks for one voxel's correlation data
input: the voxel, the number of subjects
output: update values to the voxel's correlation data
*****************************************/
void PreprocessOneVoxelsAnalysisData(Voxel* voxel, int step_id, int nSubs)
{
  int nTrials = voxel->nTrials;
  int row = voxel->nVoxels;
  int nPerSub=nTrials/nSubs;
  float (*mat)[row] = (float(*)[row])&(voxel->corr_vecs[step_id*nTrials*row]);
  #pragma omp parallel for
  #pragma simd
  for (int j=0; j<row; j++)
  {
    for(int i = 0 ; i < nSubs ; i++)
    {
      float mean = 0.0f;
    	float std_dev = 0.0f;
    	for(int k = i*nPerSub; k < (i+1)*nPerSub; k++)
    	{
    	  float num = 1.0f + mat[k][j]; 
     	  float den = 1.0f - mat[k][j];
     	  num = (num <= 0.0f) ? TINYNUM : num;
     	  den = (den <= 0.0f) ? TINYNUM : den;
     	  mat[k][j] = 0.5f * logf(num/den);
     	  mean += mat[k][j];
    	  std_dev += mat[k][j] * mat[k][j];
     	}
      mean = mean / (float)nPerSub;
    	std_dev = std_dev / (float)nPerSub - mean*mean;
      float inv_std_dev = (std_dev <= 0.0f) ? 0.0f : 1.0f / sqrt(std_dev);
    	for(int k = i*nPerSub ; k < (i+1)*nPerSub ; k++)
     	{
     	  mat[k][j] = (mat[k][j] - mean) * inv_std_dev;
     	}
    }
  }
  return;
}

/****************************************
Fisher transform the correlation values (coefficients) then z-scored them across within-subject blocks for all voxels' correlation data
input: the voxel array, the length of this array (the number of voxels, "step"), the number of subjects
output: update values to the voxels' correlation data
*****************************************/
void PreprocessAllVoxelsAnalysisData(Voxel* voxels, int step, int nSubs)
{    
  int i;
  //#pragma omp parallel for private(i)
  for (i=0; i<step; i++)
  {
    PreprocessOneVoxelsAnalysisData(voxels, i, nSubs);
  }
  return;
}

void PreprocessAllVoxelsAnalysisData_flat(Voxel* voxels, int step, int nSubs)
{
  int nTrials = voxels->nTrials;
  int row = voxels->nVoxels;
  int nPerSub=nTrials/nSubs;
  #pragma omp parallel for
  for(int v = 0 ; v < step*nSubs ; v++)
  {
    int s = v % nSubs;  // subject id
    int i = v / nSubs;  // voxel id
    float (*mat)[row] = (float(*)[row])&(voxels->corr_vecs[i*nTrials*row]);
    #pragma simd
    for(int j = 0 ; j < row ; j++)
    {
      float mean = 0.0f;
    	float std_dev = 0.0f;
      for(int b = s*nPerSub; b < (s+1)*nPerSub; b++)
      {
#ifdef __MIC__
        _mm_prefetch((char*)&(mat[b][j+32]), _MM_HINT_ET1);
        _mm_prefetch((char*)&(mat[b][j+16]), _MM_HINT_T0);
#endif
        float num = 1.0f + mat[b][j]; 
      	float den = 1.0f - mat[b][j];
      	num = (num <= 0.0f) ? 1e-4 : num;
      	den = (den <= 0.0f) ? 1e-4 : den;
      	mat[b][j] = 0.5f * logf(num/den);
      	mean += mat[b][j];
      	std_dev += mat[b][j] * mat[b][j];
      }
      mean = mean / (float)nPerSub;
      std_dev = std_dev / (float)nPerSub - mean*mean;
      float inv_std_dev = (std_dev <= 0.0f) ? 0.0f : 1.0f / sqrt(std_dev);
      for(int b = s*nPerSub; b < (s+1)*nPerSub; b++)
      {
        mat[b][j] = (mat[b][j] - mean) * inv_std_dev;
      }
    }
  }
}
