/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"
#include "LibSVM.h"

int SVMPredictCorrelationWithMasks(RawMatrix** r_matrices1, RawMatrix** r_matrices2, int nSubs, const char* maskFile1, const char* maskFile2, int nTrials, Trial* trials, int nTests, int is_quiet_mode);
float* GetPartialInnerSimMatrixWithMasks(int nSubs, int nTrials, int sr, int rowLength, Trial* trials, RawMatrix** masked_matrices1, RawMatrix** masked_matrices2);
int SVMPredictActivationWithMasks(RawMatrix** avg_matrices, int nSubs, const char* maskFile, int nTrials, Trial* trials, int nTests, int is_quiet_mode);
