/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"
#include "LibSVM.h"
#include "new_svm.h"
#define MBLK  16
#define NBLK  9
#define KBLK  96

VoxelScore* GetSVMPerformance(int me, CorrMatrix** c_matrices, int nTrainings, int nFolds);
void print_null(const char* s);
SVMProblem* GetSVMProblem(CorrMatrix** c_matrices, int row, int startIndex, int nTrainings);
SVMProblem* GetSVMProblemWithPreKernel(CorrMatrix** c_matrices, int row, int startIndex, int nTrainings);
float DoSVM(int nFolds, SVMProblem* prob, SVMParameter* param);
float DOSVMNew(float* data, int nPoints, int nDimension, int nFolds, float* labels, int vid);
VoxelScore* GetVoxelwiseSVMPerformance(int me, Trial* trials, Voxel* voxels, int step, int nTrainings, int nFolds);
VoxelScore* GetVoxelwiseNewSVMPerformance(int me, Trial* trials, Voxel* voxels, int step, int nTrainings, int nFolds);
SVMProblem* GetSVMProblemWithPreKernel2(Trial* trials, Voxel* voxel, int step_id, int row, int nTrainings);
void GetNewSVMProblemWithPreKernel(Trial* trials, Voxel* voxel, int step_id, int row, int nTrainings, float** p_data, float** p_labels);
void ComputeSimMatrix(float* corr_vec, int row, int col, float* simMatrix);
void custom_ssyrk(const MKL_INT M, const MKL_INT K, float *A, const MKL_INT lda, float *C, const MKL_INT ldc);
void sgemm_assembly(float* A, float* B, float* C, float* A_prefetch = NULL, float* B_prefetch = NULL, float* C_prefetch = NULL);
