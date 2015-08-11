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
Voxel* ComputeAllVoxelsAnalysisData(Voxel* voxels, Trial* trials, int nTrials, int nSubs, int nTrainings, int sr, int step, TrialData* td1, TrialData* td2)
{
  using std::cout;
  using std::endl;
#if __MEASURE_TIME__
  float t=0.0f, t_corr=0.0f;
  struct timeval start, end;
#endif
  for (size_t i=0; i<step; i++)
  {
    voxels->vid[i]=sr+i;
  }
//float t;
//struct timeval start, end;
//gettimeofday(&start, 0);
  size_t row=td2->nVoxels;    // assuming all matrices have the same size
  size_t ml=trials[0].ec-trials[0].sc+1;    // assuming all blocks have the same size
  int nPerSubj=nTrials/nSubs;    // assuming all subjects have the same number of blocks
  int m_max = (step/BLK2)*BLK2;
  memset((void*)voxels->kernel_matrices, 0, sizeof(float)*nTrainings*nTrainings*step);
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
#if __MEASURE_TIME__
      gettimeofday(&start, 0);
#endif
      //cout<<td1->scs[0]<<" "<<td1->scs[10]<<endl<<flush;
      sgemmTransposeMerge(td1, td2, BLK2, row, ml, nPerSubj, nSubs, voxels->corr_vecs, row*nTrials, sr+m);
#if __MEASURE_TIME__
      gettimeofday(&end, 0);
      t_corr+=end.tv_sec-start.tv_sec+(end.tv_usec-start.tv_usec)*0.000001;
#endif
#ifdef __MIC__
#if __MEASURE_TIME__
      gettimeofday(&start, 0);
#endif
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
           (const int)k_range, voxels->corr_vecs+prob_ID*row*nTrials + start_k,
           (const int)row, voxels->kernel_matrices+(prob_ID+m)*nTrainings*nTrainings, Clocks[prob_ID], (const int)nTrainings);
      }
#if __MEASURE_TIME__
      gettimeofday(&end, 0);
      t+=end.tv_sec-start.tv_sec+(end.tv_usec-start.tv_usec)*0.000001;
#endif
#else
#if __MEASURE_TIME__
      gettimeofday(&start, 0);
#endif
      #pragma omp parallel for
      for (int mm=m; mm<m+BLK2; mm++)
      {
        //custom_ssyrk_old((const int)nTrainings, (const int)row, voxels->corr_output+(mm-m)*row*nTrials, (const int)row, voxels->corr_vecs+mm*nTrainings*nTrainings, (const int)nTrainings);
        cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, nTrainings, row, 1.0, voxels->corr_vecs+(mm-m)*row*nTrials, row, 0.0, voxels->kernel_matrices+mm*nTrainings*nTrainings, nTrainings);
      }
#if __MEASURE_TIME__
      gettimeofday(&end, 0);
      t+=end.tv_sec-start.tv_sec+(end.tv_usec-start.tv_usec)*0.000001;
#endif
#endif
    }
    else
    {
#if __MEASURE_TIME__
      gettimeofday(&start, 0);
#endif
      sgemmTransposeMerge(td1, td2, step-m_max, row, ml, nPerSubj, nSubs, voxels->corr_vecs, row*nTrials, sr+m);
#if __MEASURE_TIME__
      gettimeofday(&end, 0);
      t_corr+=end.tv_sec-start.tv_sec+(end.tv_usec-start.tv_usec)*0.000001;
#endif
#ifdef __MIC__
#if __MEASURE_TIME__
      gettimeofday(&start, 0);
#endif
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
        custom_ssyrk((const int)nTrainings, 
           (const int)k_range, voxels->corr_vecs+prob_ID*row*nTrials + start_k,
           (const int)row, voxels->kernel_matrices+(prob_ID+m)*nTrainings*nTrainings, Clocks[prob_ID], (const int)nTrainings);
      }
#if __MEASURE_TIME__
      gettimeofday(&end, 0);
      t+=end.tv_sec-start.tv_sec+(end.tv_usec-start.tv_usec)*0.000001;
#endif
#else
#if __MEASURE_TIME__
      gettimeofday(&start, 0);
#endif
      #pragma omp parallel for
      for (int mm=m; mm<step; mm++)
      {
        cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, nTrainings, row, 1.0, voxels->corr_vecs+(mm-m)*row*nTrials, row, 0.0, voxels->kernel_matrices+mm*nTrainings*nTrainings, nTrainings);
      }
#if __MEASURE_TIME__
      gettimeofday(&end, 0);
      t+=end.tv_sec-start.tv_sec+(end.tv_usec-start.tv_usec)*0.000001;
#endif
#endif
    } // else m
  } //for m
#if __MEASURE_TIME__
  cout<<"correlation computing and normalization time: "<<t_corr<<"s"<<endl;
  cout<<"kernel computing time: "<<t<<"s"<<endl;
#endif
  return voxels;
}

VoxelScore* GetVoxelwiseCorrVecSum(int me, Voxel* voxels, int step, int sr, TrialData* td1, TrialData* td2)  // step is the number of voxels
{
  if (me==0)  //sanity check
  {
    FATAL("the master node isn't supposed to do classification jobs");
  }
  int nTrials = voxels->nTrials;
  int i;
  for (i=0; i<step; i++)
  {
    voxels->vid[i]=sr+i;
  }
  size_t row1=td1->nVoxels;    // assuming all matrices have the same size
  size_t row2=td2->nVoxels;    // assuming all matrices have the same size
  float* bufs1 = td1->data;
  float* bufs2 = td2->data;
  #pragma omp parallel for
  for (i=0; i<nTrials; i++)
  {
    size_t ml = td1->trialLengths[i];
    sgemmTranspose(bufs1+i*row1*ml+sr*ml, bufs2+i*row2*ml, step, row2, ml, voxels->corr_vecs+i*row2, row2*nTrials);  // assume all blocks have the same length
  }
  VoxelScore* scores = new VoxelScore[step];  // get step voxels' scores here
  #pragma omp parallel for
  for (int i=0; i<step; i++)
  {
    (scores+i)->vid = voxels->vid[i];
    (scores+i)->score = ComputeCorrVecSum(voxels, i); // compute the sum for one voxel
  }
  return scores;
}

float ComputeCorrVecSum(Voxel* voxels, int voxel_id)
{
  int nTrials = voxels->nTrials;
  size_t nVoxels = voxels->nVoxels;
  float* corr_vecs = voxels->corr_vecs+voxel_id*nTrials*nVoxels;
  float sum=0.0f;
  #pragma simd
  for (size_t i=0; i<nTrials*nVoxels; i++)
  {
    sum += std::isnan(corr_vecs[i])?0:fabs(corr_vecs[i]);
  }
  return sum/nTrials/nVoxels;
}
