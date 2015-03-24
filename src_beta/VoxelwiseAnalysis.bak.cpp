/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

// backup for the first attempt of merging kernel matrix computing

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
Voxel* ComputeAllVoxelsAnalysisData(Voxel* voxels, Trial* trials, int nTrials, int nSubs, int sr, int step, RawMatrix** matrices1, RawMatrix** matrices2, float* bufs1, float* bufs2)
{
  int i;
  //Voxel** voxels = new Voxel*[step];
  for (i=0; i<step; i++)
  {
    voxels->vid[i]=sr+i;
    //voxels[i] = new Voxel(sr+i, nTrials, matrices2[0]->row);  // assume the number of voxels (row) is the same accross blocks
    //voxels[i]->corr_vecs = (float*)_mm_malloc(sizeof(float)*nTrials*matrices2[0]->row, 64);
  }
/*  float* bufs1[nTrials];
  float* bufs2[nTrials];
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
    getBuf(sc, ec, row1, col1, matrices1[sid]->matrix, buf1);
    bufs1[i]=buf1;
    float* buf2 = (float*)_mm_malloc(sizeof(float)*row2*(ec-sc+1), 64);
    getBuf(sc, ec, row2, col2, matrices2[sid]->matrix, buf2);
    bufs2[i]=buf2;
  }*/
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
    /*for (int j=0; j<step; j++)
    {
      vectorMatMultiplyTranspose(matrices2[sid]->matrix+cur_col*row2, sizeof(float)*row2*ml, matrices1[sid]->matrix+cur_col*row1+(sr+j)*ml, sizeof(float)*ml, (voxels->corr_vecs)+j*nTrials*row2+i*row2, sizeof(float)*row2);
      //vectorMatMultiply(bufs2[i], sizeof(float)*row2*ml, bufs1[i]+(sr+j)*ml, sizeof(float)*ml, (voxels->corr_vecs)+j*nTrials*row2+i*row2, sizeof(float)*row2);
      //cblas_sgemv(CblasRowMajor, CblasNoTrans, row2, ml, 1.0, bufs2[i], ml, bufs1[i]+(sr+j)*ml, 1, 0.0, (voxels[j]->corr_vecs)+i*row2, 1);
    }*/
  }
#else
  int row=matrices1[0]->row;    // assuming all matrices have the same size
  int ml=trials[0].ec;    // assuming all blocks have the same size
  int nPerSubj=nTrials/nSubs;    // assuming all subjects have the same number of blocks
  for (int i=0; i<nTrials*nTrials*step; i++)
  {
    voxels->corr_vecs[i] = 0.0f;
  }
  sgemmTransposeMerge(bufs1+sr*ml, bufs2, step, row, ml, nPerSubj, nSubs, nTrials, voxels->corr_vecs, row*nTrials, trials);
#endif
/*
  int tpp = 10;
  #pragma omp parallel for schedule(dynamic)
  for (int ii=0; ii<tpp*nTrials; ii++)
  {
    int i = ii / tpp;
    int j = ii % tpp;
    int cur_col = trials[i].sc;
    int ml = trials[i].ec;
    int sid = trials[i].sid;
    int row1 = matrices1[sid]->row;
    int col1 = matrices1[sid]->col;
    int row2 = matrices2[sid]->row;
    int col2 = matrices2[sid]->col;
    int blksize = (row2 + tpp - 1) / tpp;
    int start = blksize * j;
    int end = blksize * (j+1);
    if(end > row2) end = row2;
    int len = end-start;
    //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, step, row2, ml, 1.0, bufs1[i]+sr*ml, ml, bufs2[i], ml, 0.0, voxels->corr_vecs+i*row2, row2*nTrials);
    //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, step, row2, ml, 1.0, matrices1[sid]->matrix+cur_col*row1+sr*ml, ml, matrices2[sid]->matrix+cur_col*row2, ml, 0.0, voxels->corr_vecs+i*row2, row2*nTrials);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, step, len, ml, 1.0, matrices1[sid]->matrix+cur_col*row1+sr*ml, ml, matrices2[sid]->matrix+cur_col*row2 + start*ml, ml, 0.0, voxels->corr_vecs+i*row2+start, row2*nTrials);
  }*/
/*
  for (i=0; i<nTrials; i++)
  {
    int sc = trials[i].sc;
    int ec = trials[i].ec;
    int sid = trials[i].sid;
    int row1 = matrices1[sid]->row;
    int col1 = matrices1[sid]->col;
    int row2 = matrices2[sid]->row;
    int col2 = matrices2[sid]->col;
    vectorMatMultiply2(bufs2[i], sizeof(float)*row2*ml, bufs1[i]+sr*ml, sizeof(float)*ml, voxels, step, i);
  }
*/
/*
  for (i=0; i<step; i++)
  {
    int row1 = matrices1[0]->row;
    int col1 = matrices1[0]->col;
    int row2 = matrices2[0]->row;
    int col2 = matrices2[0]->col;
    vectorMatMultiply3(bufs2, nTrials, sizeof(float)*row2*ml, bufs1, sizeof(float)*ml, voxels[i]->corr_vecs, i+sr);
  }
*/
//gettimeofday(&end, 0);
//t=end.tv_sec-start.tv_sec+(end.tv_usec-start.tv_usec)*0.000001;
//printf("computing time %f\n", t);
  /*for (i=0; i<nTrials; i++)
  {
    _mm_free(bufs1[i]);
    _mm_free(bufs2[i]);
  }*/
  return voxels;
}

#if 0
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
#endif

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

// total flops: ((4+logf)*34470*216+(64+sqrt)*34470*18)*120
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

/****************************************
Perform vector matrix multiply for fine-grained threads
input: matrix data, matrix data size, vector data, vector data size, output data location, output data size (all sizes are in bytes)
output: update values to the output data
*****************************************/
void vectorMatMultiply(float* mat, int mat_size, float* vec, int vec_size, float* output, int output_size)
{
  int col = vec_size/sizeof(float);
  int row = mat_size/sizeof(float)/col;
  #pragma vector nontemporal (output)
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
#define BLK 16  // for N
#define BLK2 5 // for M
#define COL 12

void vectorMatMultiplyTranspose(float* mat, int mat_size, float* vec, int vec_size, float* output, int output_size)
{
  int col = vec_size/sizeof(float);
  int row = mat_size/sizeof(float)/col;
  int row_max = (row / BLK) * BLK;
  float mat_T[COL*BLK];
  for (int i=0 ;i<row; i++)
  {
    output[i] = 0.0f;
  }
  for(int r=0; r<row; r+=BLK)
  {
    // transpose
    if(r < row_max)
    {
      for(int cc=0; cc<COL; cc++)
      {
        for(int rr=0; rr<BLK; rr++)
        {
          mat_T[cc*BLK+rr] = mat[cc+(r+rr)*COL];
        }
      }
      for (int i=0; i<COL; i++)
      {
        for (int j=0; j<BLK; j++)
        {
          output[r+j] += vec[i]*mat_T[i*BLK+j];
        }
      }
    }
    else  // last block
    {
      for(int cc=0; cc<COL; cc++)
      {
        for(int rr=0; rr<row-row_max; rr++)
        {
          mat_T[cc*BLK+rr] = mat[cc+(r+rr)*COL];
        }
      }
      for (int i=0; i<COL; i++)
      {
        for (int j=0; j<row-row_max; j++)
        {
          output[r+j] += vec[i]*mat_T[i*BLK+j];
        }
      }
    }
  }
  return;
}

//e.g. M=120, N=34470, K=12, mat1 M*K, mat2 N*K
void sgemmTranspose(float* mat1, float* mat2, const MKL_INT M, const MKL_INT N, const MKL_INT K, float* output, const MKL_INT ldc)
{
  int m_max = (M / BLK2) * BLK2;
  int n_max = (N / BLK) * BLK;
  float mat_T[K*BLK];
  float output_local[BLK2*BLK];
  for(int r=0; r<N; r+=BLK)
  {
    if(r < n_max)
    {
      // transpose
      for(int cc=0; cc<K; cc++)
      {
        for(int rr=0; rr<BLK; rr++)
        {
          mat_T[cc*BLK+rr] = mat2[cc+(r+rr)*K];
        }
      }
      for (int m=0; m<M; m+=BLK2)
      {
        for (int i=0; i<BLK2*BLK; i++)
        {
          output_local[i] = 0.0f;
        }
        if (m < m_max)
        {
          for (int i=0; i<BLK2; i++)
          {
            for (int j=0; j<BLK; j++)
            {
              for (int k=0; k<K; k++)
              {
                output_local[i*BLK+j] += mat1[(m+i)*K+k]*mat_T[k*BLK+j];
              }
            }
          }
          for (int i=0; i<BLK2; i++)
          {
            #pragma vector nontemporal
            for (int j=0; j<BLK; j++)
            {
              output[(m+i)*ldc+r+j] = output_local[i*BLK+j];
            }
          }
        } //if m
        else  //last block
        {
          for (int i=0; i<M-m_max; i++)
          {
            for (int j=0; j<BLK; j++)
            {
              for (int k=0; k<K; k++)
              {
                output_local[i*BLK+j] += mat1[(m+i)*K+k]*mat_T[k*BLK+j];
              }
            }
          }
          for (int i=0; i<M-m_max; i++)
          {
            #pragma vector nontemporal
            for (int j=0; j<BLK; j++)
            {
              output[(m+i)*ldc+r+j] = output_local[i*BLK+j];
            }
          }
        } //else m
      } //for m
    } //if r
    else  // last block
    {
      for(int cc=0; cc<K; cc++)
      {
        for(int rr=0; rr<N-n_max; rr++)
        {
          mat_T[cc*BLK+rr] = mat2[cc+(r+rr)*K];
        }
      }
      for (int m=0; m<M; m+=BLK2)
      {
        for (int i=0; i<BLK2*BLK; i++)
        {
          output_local[i] = 0.0f;
        }
        if (m < m_max)
        {
          for (int i=0; i<BLK2; i++)
          {
            for (int j=0; j<N-n_max; j++)
            {
              for (int k=0; k<K; k++)
              {
                output_local[i*BLK+j] += mat1[(m+i)*K+k]*mat_T[k*BLK+j];
              }
            }
          }
          for (int i=0; i<BLK2; i++)
          {
            #pragma vector nontemporal
            for (int j=0; j<N-n_max; j++)
            {
              output[(m+i)*ldc+r+j] = output_local[i*BLK+j];
            }
          }
        } //if m
        else  //last block
        {
          for (int i=0; i<M-m_max; i++)
          {
            for (int j=0; j<N-n_max; j++)
            {
              for (int k=0; k<K; k++)
              {
                output_local[i*BLK+j] += mat1[(m+i)*K+k]*mat_T[k*BLK+j];
              }
            }
          }
          for (int i=0; i<M-m_max; i++)
          {
            #pragma vector nontemporal
            for (int j=0; j<N-n_max; j++)
            {
              output[(m+i)*ldc+r+j] = output_local[i*BLK+j];
            }
          }
        } //else m
      } //for m
    } //else r
  } //for r
  return;
}

/****************************************
Perform vector matrix multiply for fine-grained threads
input: matrix data, matrix data size, vector data, vector data size, output data location, output data size (all sizes are in bytes)
output: update values to the output data
*****************************************/
void vectorMatMultiply2(float* mat, int mat_size, float* vec, int vec_size, Voxel** voxels, int step, int trialId)
{
  int col = vec_size/sizeof(float);
  int row = mat_size/sizeof(float)/col;
  #pragma omp parallel for
  for (int i=0; i<row; i++)
  {
    for (int k=0; k<step; k++)
    {
      float sum=0.0;  // use an additional variable for reduction in the following vectorization
      #pragma loop count(12)
      for (int j=0; j<col; j++)
      {
        sum += mat[i*col+j]*vec[k*col+j];
      }
      *(voxels[k]->corr_vecs+trialId*row+i)=sum;
      //*(voxels[0]->corr_vecs+trialId*row+i)=sum;
    }
  }
  return;
}

void vectorMatMultiply3(float* bufs2[], int nTrials, int mat_size, float* bufs1[], int vec_size, float* output, int vid)
{
  int col = vec_size/sizeof(float);
  int row = mat_size/sizeof(float)/col;
  #pragma omp parallel for
  for (int k=0; k<nTrials; k++)
  {
    float* mat=bufs2[k];
    float* vec=bufs1[k]+vid*col;
    #pragma vector nontemporal (output)
    for (int i=0; i<row; i++)
    {
      float sum=0.0;  // use an additional variable for reduction in the following vectorization
      #pragma loop count(12)
      for (int j=0; j<col; j++)
      {
        sum += mat[i*col+j]*vec[j];
      }
      output[k*row+i]=sum;
    }
  }
  return;
}

// merge correlation computing and normalization together
void sgemmTransposeMerge(float* mat1, float* mat2, const MKL_INT M, const MKL_INT N, const MKL_INT K, const int nPerSubj, const int nSubjs, const int nTrials, float* output, const MKL_INT ldc, Trial* trials)
{
  int m_max = (M / BLK2) * BLK2;
  int n_max = (N / BLK) * BLK;
  #pragma omp parallel for// collapse(2)// schedule(dynamic)
  //for (int s=0; s<nSubjs; s++)
  //{
    for(int n=0; n<N; n+=BLK)
    {
      float mat_T[K*BLK*nTrials];
      float output_local[BLK2*BLK*nTrials];
      if(n < n_max)
      {
        // transpose
        for (int ss=0; ss<nTrials; ss++)
        {
          int s = ss/nPerSubj;
          int cur_col = trials[ss].sc;
          for(int cc=0; cc<K; cc++)
          {
            for(int rr=0; rr<BLK; rr++)
            {
              mat_T[ss*BLK*K+cc*BLK+rr] = mat2[s*nPerSubj*K*N+cur_col*N+cc+(n+rr)*K];
            }
          }
        }
        for (int m=0; m<M; m+=BLK2)
        {
          for (int ss=0; ss<nTrials; ss++)
          {
            int s = ss/nPerSubj;
            for (int i=0; i<BLK2*BLK; i++)
            {
              output_local[ss*BLK2*BLK+i] = 0.0f;
            }
            int cur_col = trials[ss].sc;
            if (m < m_max)
            {
              for (int i=0; i<BLK2; i++)
              {
                for (int j=0; j<BLK; j++)
                {
                  for (int k=0; k<K; k++)
                  {
                    output_local[ss*BLK2*BLK+i*BLK+j] += mat1[s*nPerSubj*K*N+cur_col*N+(m+i)*K+k]*mat_T[ss*BLK*K+k*BLK+j];
                  }
                }
              }
            } //if m
            else  //last block
            {
              for (int i=0; i<M-m_max; i++)
              {
                for (int j=0; j<BLK; j++)
                {
                  for (int k=0; k<K; k++)
                  {
                    output_local[ss*BLK2*BLK+i*BLK+j] += mat1[s*nPerSubj*K*N+cur_col*N+(m+i)*K+k]*mat_T[ss*BLK*K+k*BLK+j];
                  }
                }
              } //else m
            } // for ss
          } //for m
          if (m < m_max)
          {
            // z-scoring etc.
            NormalizeBlkData(output_local, nPerSubj, nSubjs, nTrials);
            ComputePartialKernelMatrix(output, output_local, nTrials, m, BLK2, BLK);
            /*for (int ss=0; ss<nTrials; ss++)
            {
              for (int i=0; i<BLK2; i++)
              {
                #pragma vector nontemporal
                for (int j=0; j<BLK; j++)
                {
                  output[ss*N+(m+i)*ldc+n+j] = output_local[ss*BLK2*BLK+i*BLK+j];  //m+i is vid
                }
              }
            }*/
          } //if m
          else
          {
            // z-scoring etc.
            NormalizeBlkData(output_local, nPerSubj, nSubjs, nTrials);
            ComputePartialKernelMatrix(output, output_local, nTrials, m, M-m_max, BLK);
            /*for (int ss=0; ss<nTrials; ss++)
            {
              for (int i=0; i<M-m_max; i++)
              {
                #pragma vector nontemporal
                for (int j=0; j<BLK; j++)
                {
                  output[ss*N+(m+i)*ldc+n+j] = output_local[ss*BLK2*BLK+i*BLK+j];  //m+i is vid
                }
              }
            }*/
          } //else m
        } //for m
      }// if n
      else
      {
        // transpose
        for (int ss=0; ss<nTrials; ss++)
        {
          int s = ss/nPerSubj;
          int cur_col = trials[ss].sc;
          for(int cc=0; cc<K; cc++)
          {
            for(int rr=0; rr<N-n_max; rr++)
            {
              mat_T[ss*BLK*K+cc*BLK+rr] = mat2[s*nPerSubj*K*N+cur_col*N+cc+(n+rr)*K];
            }
          }
        }
        for (int m=0; m<M; m+=BLK2)
        {
          for (int ss=0; ss<nTrials; ss++)
          {
            int s = ss/nPerSubj;
            for (int i=0; i<BLK2*BLK; i++)
            {
              output_local[ss*BLK2*BLK+i] = 0.0f;
            }
            int cur_col = trials[ss].sc;
            if (m < m_max)
            {
              for (int i=0; i<BLK2; i++)
              {
                for (int j=0; j<N-n_max; j++)
                {
                  for (int k=0; k<K; k++)
                  {
                    output_local[ss*BLK2*BLK+i*BLK+j] += mat1[s*nPerSubj*K*N+cur_col*N+(m+i)*K+k]*mat_T[ss*BLK*K+k*BLK+j];
                  }
                }
              }
            } //if m
            else  //last block
            {
              for (int i=0; i<M-m_max; i++)
              {
                for (int j=0; j<N-n_max; j++)
                {
                  for (int k=0; k<K; k++)
                  {
                    output_local[ss*BLK2*BLK+i*BLK+j] += mat1[s*nPerSubj*K*N+cur_col*N+(m+i)*K+k]*mat_T[ss*BLK*K+k*BLK+j];
                  }
                }
              }
            } //else m
          } //for ss
          if (m < m_max)
          {
            // z-scoring etc.
            NormalizeBlkData(output_local, nPerSubj, nSubjs, nTrials);
            ComputePartialKernelMatrix(output, output_local, nTrials, m, BLK2, N-n_max);
            /*for (int ss=0; ss<nTrials; ss++)
            {
              for (int i=0; i<BLK2; i++)
              {
                #pragma vector nontemporal
                for (int j=0; j<N-n_max; j++)
                {
                  output[ss*N+(m+i)*ldc+n+j] = output_local[ss*BLK2*BLK+i*BLK+j];  //m+i is vid
                }
              }
            }*/
          } //if m
          else
          {
            // z-scoring etc.
            NormalizeBlkData(output_local, nPerSubj, nSubjs, nTrials);
            ComputePartialKernelMatrix(output, output_local, nTrials, m, M-m_max, N-n_max);
            /*for (int ss=0; ss<nTrials; ss++)
            {
              for (int i=0; i<M-m_max; i++)
              {
                #pragma vector nontemporal
                for (int j=0; j<N-n_max; j++)
                {
                  output[ss*N+(m+i)*ldc+n+j] = output_local[ss*BLK2*BLK+i*BLK+j];  //m+i is vid
                }
              }
            }*/
          } //else m
        } //for m
      }// else n
    }
  //} // for s
  return;
}

// data contains nTrials number of BLK2*BLK matrices
// normalization contains two steps
// 1. Fisher-transform each value
// 2. z-score across every entry of BLK2*BLK matrices
// or one can treat data as a nTrials-row, BLK2*BLK-column matrix
// z-scoring goes across columns
void NormalizeBlkData(float* data, const int nPerSubj, const int nSubjs, const int nTrials)
{
  //__assume_aligned(data, 64);

  for (int i=0; i<nSubjs; i++)
  {
  #pragma simd
  for(int j=0; j<BLK2*BLK; j++)
  {
    float mean = 0.0f;
  	float std_dev = 0.0f;
    for(int b=i*nPerSubj; b<(i+1)*nPerSubj; b++)
    {
#ifdef __MIC__
      _mm_prefetch((char*)&(data[b*BLK2*BLK+j+16]), _MM_HINT_T0);
      //_mm_prefetch((char*)&(data[b*BLK2*BLK+j+32]), _MM_HINT_ET1);
#endif
      float num = 1.0f + data[b*BLK2*BLK+j];
    	float den = 1.0f - data[b*BLK2*BLK+j];
    	num = (num <= 0.0f) ? 1e-4 : num;
     	den = (den <= 0.0f) ? 1e-4 : den;
     	data[b*BLK2*BLK+j] = 0.5f * logf(num/den);
     	mean += data[b*BLK2*BLK+j];
     	std_dev += data[b*BLK2*BLK+j] * data[b*BLK2*BLK+j];
    }
    mean = mean / (float)nPerSubj;
    std_dev = std_dev / (float)nPerSubj - mean*mean;
    float inv_std_dev = (std_dev <= 0.0f) ? 0.0f : 1.0f / sqrt(std_dev);
    for(int b=i*nPerSubj; b<(i+1)*nPerSubj; b++)
    {
      data[b*BLK2*BLK+j] = (data[b*BLK2*BLK+j] - mean) * inv_std_dev;
    }
  }
  }
}

// for BLK2 voxels, compute BLK2 partial kernel matrices
// kernel matrix: nTrials*nTrials
// corr matrix: nTrials-row, BLK*BLK2-column
void ComputePartialKernelMatrix(float* kernel_matrix, float* corr_matrix, const int nTrials, const int m, const int m_blk, const int n_blk)
{
  float mat_T[nTrials*BLK];
  // m_blk voxels
  for (int v=0; v<m_blk; v++)
  {
    float output_local[nTrials*nTrials];
    for (int ii=0; ii<nTrials*nTrials; ii++)
    {
      output_local[ii]=0.0f;
    }
    // transpose
    for(int cc=0; cc<n_blk; cc++)
    {
      for(int rr=0; rr<nTrials; rr++)
      {
        mat_T[cc*nTrials+rr] = corr_matrix[rr*BLK*BLK2+v*BLK+cc];
      }
    }
    for (int i=0; i<nTrials; i++)
    {
      for (int j=0; j<nTrials; j++)
      {
        for (int k=0; k<n_blk; k++)
        {
          output_local[i*nTrials+j] += corr_matrix[i*BLK*BLK2+v*BLK+k]*mat_T[k*nTrials+j];
        }
      }
    }
    /*for (int ii=0; ii<nTrials*nTrials; ii++)
    {
      #pragma omp atomic
      kernel_matrix[(m+v)*nTrials*nTrials+ii]+=output_local[ii];
    }*/
  }
}
