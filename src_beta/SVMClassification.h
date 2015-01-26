/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"
#include "LibSVM.h"

VoxelScore* GetSVMPerformance(int me, CorrMatrix** c_matrices, int nTrainings, int nFolds);
void print_null(const char* s);
SVMProblem* GetSVMProblem(CorrMatrix** c_matrices, int row, int startIndex, int nTrainings);
SVMProblem* GetSVMProblemWithPreKernel(CorrMatrix** c_matrices, int row, int startIndex, int nTrainings);
float DoSVM(int nFolds, SVMProblem* prob, SVMParameter* param);
VoxelScore* GetVoxelwiseSVMPerformance(int me, Trial* trials, Voxel* voxels, int step, int nTrainings, int nFolds);
SVMProblem* GetSVMProblemWithPreKernel2(Trial* trials, Voxel* voxel, int step_id, int row, int nTrainings);
void ComputeSimMatrix(float* corr_vec, int row, int col, float* simMatrix);
