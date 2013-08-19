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
	int i, j, delta_col=0;
	for (i=0; i<row; i++)
	{
    delta_col = 0;
		double mean=0, sd=0;  // float here is not precise enough to handle
		for (j=start_col; j<=end_col; j++)
		{
			mean += (double)mat[i*col+j];
			sd += (double)mat[i*col+j] * mat[i*col+j]; // convert to double to avoid overflow
      delta_col++;
    }
		mean /= delta_col;
		sd = sd - delta_col * mean * mean;
    //if (sd < 0) {cerr<<"sd<0! "<<sd; exit(1);}
		sd = sqrt(sd);
    //if (sd == 0) {cerr<<"sd=0!"<<endl; exit(1);}
		for (j=start_col; j<=end_col; j++)
		{
      if (sd!=0)
			  buf[i*delta_col+j-start_col] = (mat[i*col+j] - mean) / sd; // if sd is zero, a "nan" appears
      else
        buf[i*delta_col+j-start_col] = 0;
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
  c_matrix->nVoxels = row2; //
  float* corrs = new float[step*row2];
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, step, row2, ml1, 1.0, buf1+sr*ml1, ml1, buf2, ml2, 0.0, corrs, row2);
  c_matrix->matrix = corrs;
  delete buf1;
  delete buf2;
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
  for (i=0; i<nTrials; i++)
  {
    c_matrices[i] = CorrMatrixComputation(trials[i], sr, step, matrices1, matrices2);
  }
  return c_matrices;
}
