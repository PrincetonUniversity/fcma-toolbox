/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"

int getBuf(int start_col, int end_col, int row, int col, float* mat,
           float* buf);
CorrMatrix* CorrMatrixComputation(Trial trial, int sr, int step,
                                  RawMatrix** matrices1, RawMatrix** matrices2);
CorrMatrix** ComputeAllTrialsCorrMatrices(Trial* trials, int nTrials, int sr,
                                          int step, RawMatrix** matrices1,
                                          RawMatrix** matrices2);
void matmul(float* a, float* b, float* c, int n1, int n2, int n3);
