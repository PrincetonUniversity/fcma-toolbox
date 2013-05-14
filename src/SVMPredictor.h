#include "common.h"
#include "LibSVM.h"

void SVMPredict(RawMatrix** r_matrices, RawMatrix** avg_matrices, int nSubs, int nTrials, Trial* trials, int nTests, int taskType, const char* topVoxelFile, const char* mask_file);
void CorrelationBasedClassification(int* tops, int nSubs, int nTrials, Trial* trials, int nTests, RawMatrix** r_matrices);
void ActivationBasedClassification(int* tops, int nTrials, Trial* trials, int nTests, RawMatrix** avg_matrices);
VoxelScore* ReadTopVoxelFile(const char* file, int n);
void RearrangeMatrix(RawMatrix** r_matrices, VoxelScore* scores, int row, int col, int nSubs);
float* GetInnerSimMatrix(int row, int col, int nSubs, int nTrials, Trial* trials, RawMatrix** r_matrices);
float* GetPartialInnerSimMatrix(int row, int col, int nSubs, int nTrials, int sr, int rowLength, Trial* trials, RawMatrix** r_matrices);
void GetDotProductUsingMatMul(float* simMatrix, float* values, int nTrials, int nVoxels, int lengthPerCorrVector);
void NormalizeCorrValues(float* values, int nTrials, int nVoxels, int lengthPerCorrVector, int nSubs);
SVMProblem* GetSVMTrainingSet(float* simMatrix, int nTrials, Trial* trials, int nTrainings);
