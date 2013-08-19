/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
extern "C" {
#include <cblas.h>
}
#endif

int getBuf(int start_col, int end_col, int row, int col, float* mat, float* buf);
CorrMatrix* CorrMatrixComputation(Trial trial, int sr, int step, RawMatrix** matrices1, RawMatrix** matrices2);
CorrMatrix** ComputeAllTrialsCorrMatrices(Trial* trials, int nTrials, int sr, int step, RawMatrix** matrices1, RawMatrix** matrices2);
