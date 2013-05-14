#include "common.h"

VoxelScore* GetCorrVecSum(int me, CorrMatrix** c_matrices, int nTrials);
float AllTrialsCorrVecSum(int nTrials, int startIndex, CorrMatrix** c_matrices, int length);
float GetVectorSum(float* v, int length);
