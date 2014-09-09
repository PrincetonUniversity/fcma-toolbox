/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include <cstring>
#include "common.h"
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
extern "C" {
#include <cblas.h>
}
#endif

int AlignMatrices(RawMatrix** r_matrices, int nSubs, Point* pts);
int AlignMatricesByFile(RawMatrix** r_matrices, int nSubs, const char* file, Point* pts);
void leaveSomeTrialsOut(Trial* trials, int nTrials, int tid, int nLeaveOut);
void corrMatPreprocessing(CorrMatrix** c_matrices, int n, int nSubs);
/****************************************
Fisher transform a correlation value (coefficients)
input: the correlation value
output: the transformed value
*****************************************/
inline float fisherTransformation(float v)
{
  __declspec(align(64)) float f1 = 1+v;
  if (f1<=0.0)
  {
    f1 = TINYNUM;
  }
  __declspec(align(64)) float f2 = 1-v;
  if (f2<=0.0)
  {
    f2 = TINYNUM;
  }
  return 0.5 * logf(f1/f2);
}
void z_score(float* v, int n);
RawMatrix** rawMatPreprocessing(RawMatrix** r_matrices, int n, int nTrials, Trial* trials);
float getAverage(RawMatrix* r_matrix, Trial trial, int vid);
void MatrixPermutation(RawMatrix** r_matrices, int nSubs);
