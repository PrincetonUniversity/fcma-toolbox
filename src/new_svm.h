#ifndef NEW_SVM
#define NEW_SVM
#include <float.h>
#include "common.h"
#include <mkl.h>

#define BLOCKSIZE 256
#define REDUCE0  0x00000001
#define REDUCE1  0x00000002
#define MAX_PITCH 262144
#define MAX_POINTS (MAX_PITCH/sizeof(float) - 2)
#define intDivideRoundUp(a, b) (a%b!=0)?(a/b+1):(a/b)

enum KernelType {
  NEWLINEAR,
  NEWGAUSSIAN,
  NEWPRECOMPUTED
};

struct Kernel_params{
	float gamma;
	float coef0;
	int degree;
	float b;
	std::string kernel_type;
};

enum SelectionHeuristic {FIRSTORDER, SECONDORDER, RANDOM, ADAPTIVE};

class NewCache {
 public:
  NewCache(int nPointsIn, int cacheSizeIn);
  ~NewCache();
  void findData(const int index, int &offset, bool &compute);
	void search(const int index, int &offset, bool &compute);
  void printCache();
	void printStatistics();
private:
  int nPoints;
  int cacheSize;
  class DirectoryEntry {
  public:
    enum {NEVER, EVICTED, INCACHE};
    DirectoryEntry();
    int status;
    int location;
    list<int>::iterator lruListEntry;
  };

  vector<DirectoryEntry> directory;
  list<int> lruList;
  int occupancy;
  int hits;
  int compulsoryMisses;
  int capacityMisses;
};

class Controller {
 public:
  Controller(float initialGap, SelectionHeuristic currentMethodIn, int samplingIntervalIn, int problemSize);
  void addIteration(float gap);
  void print();
  SelectionHeuristic getMethod();
 private:
  bool adaptive;
  int samplingInterval;
  vector<float> progress;
  vector<int> method;
  SelectionHeuristic currentMethod;
  vector<float> rates;
  int timeSinceInspection;
  int inspectionPeriod;
  int beginningOfEpoch;
  int middleOfEpoch;
  int currentInspectionPhase;
  float filter(int begin, int end);
  float findRate(struct timeval* start, struct timeval* finish, int beginning, int end);
  struct timeval start;
  struct timeval mid;
  struct timeval finish;
};

float crossValidationNoShuffle(float* data, int nPoints, int nDimension, int nFolds, float* labels, Kernel_params* kp, float cost, SelectionHeuristic heuristicMethod, float epsilon, float tolerance, float* transposedData, int vid);
void svmGroupClasses(int nPoints, float *labels, int **start_ret, int **count_ret, int *perm);
void performTraining(float* data, int nPoints, int nDimension, float* labels, float** p_alpha, Kernel_params* kp, float cost, SelectionHeuristic heuristicMethod, float epsilon, float tolerance, float* transposedData, int fold_id, int vid);
template<int Kernel>
void	firstOrder(float* devData, int devDataPitchInFloats, float* devTransposedData, int devTransposedDataPitchInFloats, float* devLabels, int nPoints, int nDimension, float epsilon, float cEpsilon, float* devAlpha, float* devF, float alpha1Diff, float alpha2Diff, int iLow, int iHigh, float parameterA, float parameterB, float parameterC, float* devCache, int devCachePitchInFloats, int iLowCacheIndex, int iHighCacheIndex, float* devKernelDiag, void* devResult, float cost, bool iLowCompute, bool iHighCompute, int nthreads);
void launchFirstOrder(bool iLowCompute, bool iHighCompute, int kType, int nPoints, int nDimension, float* devData, int devDataPitchInFloats, float* devTransposedData, int devTransposedDataPitchInFloats, float* devLabels, float epsilon, float cEpsilon, float* devAlpha, float* devF, float sAlpha1Diff, float sAlpha2Diff, int iLow, int iHigh, float parameterA, float parameterB, float parameterC, float* devCache, int devCachePitchInFloats, int iLowCacheIndex, int iHighCacheIndex, float* devKernelDiag, void* devResult, float cost, int nthreads);
template<int Kernel>
void	secondOrder(float* devData, int devDataPitchInFloats, float* devTransposedData, int devTransposedDataPitchInFloats, float* devLabels, int nPoints, int nDimension, float epsilon, float cEpsilon, float* devAlpha, float* devF, float alpha1Diff, float alpha2Diff, int iLow, int iHigh, float parameterA, float parameterB, float parameterC, float* devCache, int devCachePitchInFloats, int iLowCacheIndex, int iHighCacheIndex, float* devKernelDiag, void* devResult, float cost, float bHigh, bool iHighCompute, int nthreads);
void launchSecondOrder(bool iLowCompute, bool iHighCompute, int kType, int nPoints, int nDimension, float* devData, int devDataPitchInFloats, float* devTransposedData, int devTransposedDataPitchInFloats, float* devLabels, float epsilon, float cEpsilon, float* devAlpha, float* devF, float sAlpha1Diff, float sAlpha2Diff, int iLow, int iHigh, float parameterA, float parameterB, float parameterC, NewCache* kernelCache, float* devCache, int devCachePitchInFloats, int iLowCacheIndex, int iHighCacheIndex, float* devKernelDiag, void* devResult, float cost, int iteration, int nthreads);
template<int Kernel>
void initializeArrays(float* devData, int devDataPitchInFloats, float* devCache, int devCachePitchInFloats, int nPoints, int nDimension, float parameterA, float parameterB, float parameterC, float* devKernelDiag, float* devAlpha, float* devF, float* devLabels, int nthreads);
void launchInitialization(float* devData, int devDataPitchInFloats, float* devCache, int devCachePitchInFloats, int nPoints, int nDimension, int kType, float parameterA, float parameterB, float parameterC, float* devKernelDiag, float* devAlpha, float* devF, float* devLabels, int nthreads);
template<int Kernel>
void takeFirstStep(void* devResult, float* devKernelDiag, float* devData, int devDataPitchInFloats, float* devCache, int devCachePitchInFloats, float* devAlpha, float cost, int nDimension, int iLow, int iHigh, float parameterA, float parameterB, float parameterC);
void launchTakeFirstStep(void* devResult, float* devKernelDiag, float* devData, int devDataPitchInFloats, float* devCache, int devCachePitchInFloats, float* devAlpha, float cost, int nDimension, int iLow, int iHigh, int kType, float parameterA, float parameterB, float parameterC, int nthreads);
void performClassification(float *data, int nData, float *supportVectors, int nSV, int nDimension, float* alpha, Kernel_params* kp, float** p_result);
void computeKernels(float* devNorms, int devNormsPitchInFloats, float* devAlpha, int nPoints, int nSV, const KernelType kType, float coef0, int degree, float b, float* devResult);
float kernel(const float v, const float coef0, const int degree, const KernelType kType);
void makeSelfDots(float* devSource, int devSourcePitchInFloats, float* devDest, int sourceCount, int sourceLength);
void makeDots(float* devDots, int devDotsPitchInFloats, float* devSVDots, float* devDataDots, int nSV, int nPoints);
#endif
