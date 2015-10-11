/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"
#include "LibSVM.h"
#include "phisvm.h"

void SVMPredict(RawMatrix** r_matrices, RawMatrix** r_matrices2,
                RawMatrix** avg_matrices, int nSubs, int nTrials, Trial* trials,
                int nTests, Task taskType, const char* topVoxelFile,
                const char* mask_file, int is_quiet_mode);
void CorrelationBasedClassification(int* tops, int ntops, int nSubs,
                                    int nTrials, Trial* trials, int nTests,
                                    RawMatrix** r_matrices1,
                                    RawMatrix** r_matrices2, int is_quiet_mode);
void ActivationBasedClassification(int* tops, int ntops, int nTrials,
                                   Trial* trials, int nTests,
                                   RawMatrix** avg_matrices, int is_quiet_mode);
VoxelScore* ReadTopVoxelFile(const char* file, int n);
void RearrangeMatrix(RawMatrix** r_matrices, VoxelScore* scores, int row,
                     int col, int nSubs);
float* GetInnerSimMatrix(int row, int col, int nSubs, int nTrials,
                         Trial* trials, RawMatrix** r_matrices1,
                         RawMatrix** r_matrices2);
float* GetPartialInnerSimMatrix(int row, int col, int nSubs, int nTrials,
                                int sr, int rowLength, Trial* trials,
                                RawMatrix** r_matrices1,
                                RawMatrix** r_matrices2);
void GetDotProductUsingMatMul(float* simMatrix, float* values, int nTrials,
                              int nVoxels, int lengthPerCorrVector);
void NormalizeCorrValues(float* values, int nTrials, int nVoxels,
                         int lengthPerCorrVector, int nSubs);
SVMProblem* GetSVMTrainingSet(float* simMatrix, int nTrials, Trial* trials,
                              int nTrainings);
