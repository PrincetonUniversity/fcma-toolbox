/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"

Voxel* ComputeAllVoxelsAnalysisData(Voxel* voxels, Trial* trials, int nTrials,
                                    int nSubs, int nTrainings, int sr, int step,
                                    TrialData* td1, TrialData* td2);
void PreprocessAllVoxelsAnalysisData(Voxel* voxels, int step, int nSubs);
void PreprocessAllVoxelsAnalysisData_flat(Voxel* voxels, int step, int nSubs);
void PreprocessOneVoxelsAnalysisData(Voxel* voxel, int step_id, int nSubs);
VoxelScore* GetVoxelwiseCorrVecSum(int me, Voxel* voxels, int step, int sr,
                                   TrialData* td1, TrialData* td2);
float ComputeCorrVecSum(Voxel* voxels, int voxel_id);
