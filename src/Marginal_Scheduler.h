/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"

void Marginal_Scheduler(int me, int nprocs, int step, RawMatrix** r_matrices, int taskType, Trial* trials, int nTrials, int nHolds, int nSubs, int nFolds, const char* output_file, const char* mask_file1);
bool cmp1(VoxelScore w1, VoxelScore w2);
float* prepare_data(RawMatrix** masked_matrices, Trial* trials, int row, int nTrialsUsed, int nBlocksPerSub);
VoxelScore* compute_first_order(RawMatrix** masked_matrices, Trial* trials, int nTrials, int nHolds, int nBlocksPerSub, float* data, int* labels);
void compute_second_order(float* data, int* labels, int nTrialsUsed, int row, float* second_order, int sr, int step);
VoxelScore* compute_marginal_info(VoxelScore* scores, int topK, int row, float* second_order);
int* GatherLabels(Trial* trials, int nTrials);
void write_result(const char* output_file, int order, VoxelScore* scores, int row, const char* mask_file);
void Do_Marginal_Master(int nprocs, int step, int row, float* second_order, const char* output_file, const char* mask_file);
void Do_Marginal_Slave(int me, int masterId, float* data, int* labels, int row, int taskType, int nTrialsUsed);
