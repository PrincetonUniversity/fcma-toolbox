/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"

void Scheduler(int me, int nprocs, int step, RawMatrix** r_matrices, RawMatrix** r_matrices2, Task taskType, Trial* trials, int nTrials, int nHolds, int nSubs, int nFolds, const char* output_file, const char* mask_file1, const char* mask_file2, int shuffle, const char* permute_book_file);
bool cmp(VoxelScore w1, VoxelScore w2);
void DoMaster(int nprocs, int step, int row, const char* output_file, const char* mask_file);
void DoSlave(int me, int masterId, TrialData* td1, TrialData* td2, Task taskType, Trial* trials, int nTrials, int nHolds, int nSubs, int nFolds, int preset_step);
