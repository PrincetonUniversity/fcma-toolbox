#include "common.h"
#include "svm.h"

typedef struct svm_problem SVMProblem;
typedef struct svm_parameter SVMParameter;
typedef struct svm_node SVMNode;

void Searchlight(RawMatrix** avg_matrices, int nSubs, Trial* trials, int nTrials, int nTests, int nFolds, Point* pts, const char* topVoxelFile, const char* maskFile);
VoxelScore* GetSearchlightSVMPerformance(RawMatrix** avg_matrices, Trial* trials, int nTrials, int nTests, int nFolds, Point* pts);
SVMProblem* GetSearchlightSVMProblem(RawMatrix** avg_matrices, Trial* trials, int curVoxel, int nTrainings, Point* pts);
int* GetSphere(int voxelId, int nVoxels, Point* pts);
int GetPoint(int x, int y, int z, int nVoxels, Point* pts);
