/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include <cstring>
#include "common.h"
#include "ErrorHandling.h"
#ifdef USE_MKL
#include <mkl_cblas.h>
#elif defined __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern "C" {
#include <cblas.h>
}
#endif

int AlignMatrices(RawMatrix** r_matrices, int nSubs, VoxelXYZ* pts);
int AlignMatricesByFile(RawMatrix** r_matrices, int nSubs, const char* file, VoxelXYZ* pts);
void leaveSomeTrialsOut(Trial* trials, int nTrials, int tid, int nLeaveOut);
void corrMatPreprocessing(CorrMatrix** c_matrices, int n, int nSubs);
/****************************************
Fisher transform a correlation value (coefficients)
input: the correlation value
output: the transformed value
*****************************************/
inline float fisherTransformation(float v)
{
  ALIGNED(64) float f1 = 1+v;
  if (f1<=0.0)
  {
    f1 = TINYNUM;
  }
  ALIGNED(64) float f2 = 1-v;
  if (f2<=0.0)
  {
    f2 = TINYNUM;
  }
  return 0.5 * logf(f1/f2);
}
/***************************************
z-score the vectors
input: the vector, the length of the vector
output: write z-scored values to the vector
****************************************/
inline void z_score(float* v, int n)
{
  int i;
  ALIGNED(64) float mean=0, sd=0;  // float here is not precise enough to handle
  #pragma loop count(12)
  for (i=0; i<n; i++)
  {
    mean += v[i];
    sd += v[i] * v[i]; // float can cause sd<0, need to handle it later
  }
  mean /= n;
  sd = sd/n - mean * mean;
  ALIGNED(64) float inv_sd= sd<=0?0.0:1/sqrt(sd);  // do time-comsuming division once, sd==0 means all values are the same, sd<0 happens due to floating point number rounding error
  #pragma loop count(12)
  #pragma simd
  for (i=0; i<n; i++)
  {
    v[i] = (v[i] - mean)*inv_sd;
  }
}
RawMatrix** rawMatPreprocessing(RawMatrix** r_matrices, int n, int nTrials, Trial* trials);
float getAverage(RawMatrix* r_matrix, Trial trial, int vid);
void MatrixPermutation(RawMatrix** r_matrices, int nSubs, unsigned int seed, const char* permute_book_file);
void PreprocessMatrices(RawMatrix** matrices, Trial* buf_trials, Trial* trials, int nSubs, int nTrials);
