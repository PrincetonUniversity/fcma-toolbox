/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"

VoxelScore* GetCorrVecSum(int me, CorrMatrix** c_matrices, int nTrials);
float AllTrialsCorrVecSum(int nTrials, int startIndex, CorrMatrix** c_matrices, int length);
float GetVectorSum(float* v, int length);
