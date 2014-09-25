/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "MatComputation.h"
#include "common.h"

/*****************************
normalize the 64-bit double to 32-bit float matrix, to z-score the numbers
input: starting time point, end time point, number of voxels, number of total time points, the raw matrix, normalized matrix
output: write results to the normalized matrix, return the length of time points to be computed
******************************/
int getBuf(int start_col, int end_col, int row, int col, float* mat, float* buf)
{
  int i;
  int delta_col = end_col-start_col+1;
  //#pragma omp parallel for private(i)
  for (i=0; i<row; i++)
  {
    double mean=0, sd=0;  // float here is not precise enough to handle
    for (int j=start_col; j<=end_col; j++)
    {
      mean += (double)mat[i*col+j];
      sd += (double)mat[i*col+j] * mat[i*col+j]; // convert to double to avoid overflow
    }
    mean /= delta_col;
    sd = sd - delta_col * mean * mean;
    //if (sd < 0) {cerr<<"sd<0! "<<sd; exit(1);}
    sd = sqrt(sd);
    if (sd == 0)
    {
      memset(buf+i*delta_col, 0, sizeof(float)*delta_col);
      continue;
    }
    ALIGNED(64) float inv_sd_f=1/sd;  // do time-comsuming division once
    ALIGNED(64) float mean_f=mean;  // for vecterization
    #pragma simd
    for (int j=start_col; j<=end_col; j++)
    {
        buf[i*delta_col+j-start_col] = (mat[i*col+j] - mean_f) * inv_sd_f; // bds 1/sd; // if sd is zero, a "nan" appears
    }
  }
  return delta_col;
}

/******************************
compute a corr-matrix based on trial, starting row and step
input: a trial struct, starting row id, step (the row of the correlation matrix, whose column is row), the raw matrix struct array
output: the corr-matrix struct
*******************************/
CorrMatrix* CorrMatrixComputation(Trial trial, int sr, int step, RawMatrix** matrices1, RawMatrix** matrices2)
{
  int sid = trial.sid;
  int sc = trial.sc;
  int ec = trial.ec;
  int row1 = matrices1[sid]->row;
  int row2 = matrices2[sid]->row;
  int col = matrices1[sid]->col;  // the column of 1 and 2 should be the same, i.e. the number of TRs of a block
  float* mat1 = matrices1[sid]->matrix;
  float* mat2 = matrices2[sid]->matrix;
  float* buf1 = new float[row1*col]; // col is more than what really need, just in case
  float* buf2 = new float[row2*col]; // col is more than what really need, just in case
  int ml1 = getBuf(sc, ec, row1, col, mat1, buf1);  // get the normalized matrix, return the length of time points to be computed
  int ml2 = getBuf(sc, ec, row2, col, mat2, buf2);  // get the normalized matrix, return the length of time points to be computed, m1==m2
  CorrMatrix* c_matrix = new CorrMatrix();
  c_matrix->sid = sid;
  c_matrix->tlabel = trial.label;
  c_matrix->sr = sr;
  c_matrix->step = step;
  c_matrix->nVoxels = row2;
  float* corrs = new float[step*row2];
  //struct timeval start, end;
  //gettimeofday(&start, NULL);
  //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, step, row2, ml1, 1.0, buf1+sr*ml1, ml1, buf2, ml2, 0.0, corrs, row2);
  matmul(buf1+sr*ml1, buf2, corrs, step, ml1, row2);
  //if (sr==0 && sid==2 && sc==4) {cout<<corrs[99]<<" "<<corrs[100]<<endl; exit(1);}
  //gettimeofday(&end, NULL);
  //long secs = end.tv_sec-start.tv_sec;
  //cout<<"pure matrix computing: "<<secs<<"s"<<endl;
  c_matrix->matrix = corrs;
  delete[] buf1;
  delete[] buf2;
  return c_matrix;
}

/********************************
compute corr-matrices for the starting row and step across all trials
input: the trial struct array, number of trials, the starting row (since distributing to multi-nodes, step the raw matrix struct array
output: the corr-matrix struct array
*********************************/
CorrMatrix** ComputeAllTrialsCorrMatrices(Trial* trials, int nTrials, int sr, int step, RawMatrix** matrices1, RawMatrix** matrices2)
{
  int i;
  CorrMatrix** c_matrices = new CorrMatrix*[nTrials];
  #pragma omp parallel for private(i)
  for (i=0; i<nTrials; i++)
  {
    c_matrices[i] = CorrMatrixComputation(trials[i], sr, step, matrices1, matrices2);
  }
  return c_matrices;
}

// c=a*b', a: n1*n2, b: n3*n2, c: n1*n3
void matmul(float *a, float* b, float *c, int n1, int n2, int n3)
{
  memset((void*)c, 0, n1*n3*sizeof(float));
  int s = 500;
  for(int jj=0; jj<n3; jj+= s){
    for(int kk=0; kk<n2; kk+= s){
      for(int i=0;i<n1;i++){
        for(int j = jj; j<((jj+s)>n3?n3:(jj+s)); j++){
          float temp = 0.0;
          for(int k = kk; k<((kk+s)>n2?n2:(kk+s)); k++){
            //temp += a[i*n2+k]*b[k*n3+j];  // this is c=a*b
            temp += a[i*n2+k]*b[j*n2+k];
          }
          c[i*n3+j] += temp;
        }
      }
    }
  }
}
