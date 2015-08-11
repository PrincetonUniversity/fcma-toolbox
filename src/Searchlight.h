/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"
#include "svm.h"

typedef struct svm_problem SVMProblem;
typedef struct svm_parameter SVMParameter;
typedef struct svm_node SVMNode;

void Searchlight(RawMatrix** avg_matrices, int nSubs, Trial* trials,
                 int nTrials, int nTests, int nFolds, VoxelXYZ* pts,
                 const char* topVoxelFile, const char* maskFile, int shuffle,
                 const char* permute_book_file);
VoxelScore* GetSearchlightSVMPerformance(RawMatrix** avg_matrices,
                                         Trial* trials, int nTrials, int nTests,
                                         int nFolds, VoxelXYZ* pts);
SVMProblem* GetSearchlightSVMProblem(RawMatrix** avg_matrices, Trial* trials,
                                     int curVoxel, int nTrainings,
                                     VoxelXYZ* pts);
int* GetSphere(int voxelId, int nVoxels, VoxelXYZ* pts);
int GetPoint(int x, int y, int z, int nVoxels, VoxelXYZ* pts);
