#include "common.h"

void Scheduler(int me, int nprocs, int step, RawMatrix** r_matrices, int taskType, Trial* trials, int nTrials, int nHolds, int nSubs, int nFolds, const char* output_file, const char* mask_file1, const char* mask_file2);
bool cmp(VoxelScore w1, VoxelScore w2);
void DoMaster(int nprocs, int step, int row, const char* output_file, const char* mask_file);
void DoSlave(int me, int masterId, RawMatrix** matrices1, RawMatrix** matrices2, int taskType, Trial* trials, int nTrials, int nHolds, int nSubs, int nFolds);
