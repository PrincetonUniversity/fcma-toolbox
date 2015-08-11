/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"

VoxelScore* GetDistanceRatio(int me, CorrMatrix** c_matrices, int nTrials);
float DoDistanceRatioSmarter(int nTrainings, int startIndex,
                             CorrMatrix** c_matrices, int length);
// float GetVectorSum(float* v, int length);
