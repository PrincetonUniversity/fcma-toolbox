#include "common.h"

VoxelScore* GetDistanceRatio(int me, CorrMatrix** c_matrices, int nTrials);
float DoDistanceRatioSmarter(int nTrainings, int startIndex, CorrMatrix** c_matrices, int length);
//float GetVectorSum(float* v, int length);
