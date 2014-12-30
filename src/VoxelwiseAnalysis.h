/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"
#ifdef USE_MKL
#include <mkl.h>
#elif defined __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern "C" {
#include <cblas.h>
}
#endif

Voxel** ComputeAllVoxelsAnalysisData(Trial* trials, int nTrials, int sr, int step, RawMatrix** matrices1, RawMatrix** matrices2);
void PreprocessAllVoxelsAnalysisData(Voxel** voxels, int step, int nSubs);
void PreprocessOneVoxelsAnalysisData(Voxel* voxel, int nSubs);
Voxel* ComputeOneVoxelAnalysisData(Trial* trials, int vid, int nTrials, float** data_buf1, float** data_buf2, int row, int col);
void vectorMatMultiply(float* mat, int mat_size, float* vec, int vec_size, float* output, int output_size);
