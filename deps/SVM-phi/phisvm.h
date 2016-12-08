#ifndef PHI_SVM
#define PHI_SVM
#include <float.h>
#include <vector>
#include <list>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <sys/time.h>

// BLAS dependency
#ifdef USE_MKL
#include <mkl.h>
#else
typedef int MKL_INT;
#if defined __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern "C" {
#include <cblas.h>
}
#endif
#endif

#include <string>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/back_inserter.hpp>

#include <sstream>

using namespace std;

#define BLOCKSIZE 256
#define REDUCE0 0x00000001
#define REDUCE1 0x00000002
#define MAX_PITCH 262144
#define MAX_POINTS (MAX_PITCH / sizeof(float) - 2)
#define MAXITERATIONS 100000
#define intDivideRoundUp(a, b) (a % b != 0) ? (a / b + 1) : (a / b)

enum KernelType {
  NEWLINEAR,
  NEWGAUSSIAN,
  NEWPRECOMPUTED
};

struct Kernel_params {
  float gamma;
  float coef0;
  int degree;
  float b;
  std::string kernel_type;
};

enum SelectionHeuristic {
  FIRSTORDER,
  SECONDORDER,
  RANDOM,
  ADAPTIVE
};

class PhiSVMModel {
public:
  int nSamples;
  int nDimension;
  float epsilon;
  float bLow;
  float bHigh;
  float* alpha;
  float* f;
  float* kernelDiag;
  float* data;
  float* labels;
  PhiSVMModel() { // default constructor for training from the scratch
  }
  PhiSVMModel(int maxPoints, int maxDimension) {  // constructor for getting an existing model
    alpha = new float[maxPoints];
    f = new float[maxPoints];
    kernelDiag = new float[maxPoints];
    data = new float[maxPoints*maxDimension];
    labels = new float[maxPoints];
  }
  ~PhiSVMModel() {  // leave the data and labels to others to delete
    delete alpha;
    delete f;
    delete kernelDiag;
  }
};

class NewCache {
 public:
  NewCache(int nPointsIn, int cacheSizeIn);
  ~NewCache();
  void findData(const int index, int& offset, bool& compute);
  void search(const int index, int& offset, bool& compute);
  void printCache();
  void printStatistics();

 private:
  int nPoints;
  int cacheSize;
  class DirectoryEntry {
   public:
    enum {
      NEVER,
      EVICTED,
      INCACHE
    };
    DirectoryEntry();
    int status;
    int location;
    std::list<int>::iterator lruListEntry;
  };

  std::vector<DirectoryEntry> directory;
  std::list<int> lruList;
  int occupancy;
  int hits;
  int compulsoryMisses;
  int capacityMisses;
};

class Controller {
 public:
  Controller(float initialGap, SelectionHeuristic currentMethodIn,
             int samplingIntervalIn, int problemSize);
  void addIteration(float gap);
  void print();
  SelectionHeuristic getMethod();

 private:
  bool adaptive;
  int samplingInterval;
  std::vector<float> progress;
  std::vector<int> method;
  SelectionHeuristic currentMethod;
  std::vector<float> rates;
  int timeSinceInspection;
  int inspectionPeriod;
  int beginningOfEpoch;
  int middleOfEpoch;
  int currentInspectionPhase;
  float filter(int begin, int end);
  float findRate(struct timeval* start, struct timeval* finish, int beginning,
                 int end);
  struct timeval start;
  struct timeval mid;
  struct timeval finish;
};

bool hasTwoTypes(float* labels, int n);

float crossValidation(float* data, int nPoints, int nDimension,
                               int nFolds, float* labels, Kernel_params* kp,
                               float cost, SelectionHeuristic heuristicMethod,
                               float epsilon, float tolerance,
                               float* transposedData, bool shuffle);

float incrementalCrossValidation(PhiSVMModel** models, float* data, int nPoints, int nDimension,
                               int nFolds, float* labels, Kernel_params* kp,
                               float cost, SelectionHeuristic heuristicMethod,
                               float epsilon, float tolerance,
                               float* transposedData, int maxNPoints);

void svmGroupClasses(int nPoints, float* labels, int** start_ret,
                     int** count_ret, int* perm);
PhiSVMModel* performTraining(float* data, int nPoints, int nDimension, float* labels,
                     Kernel_params* kp, float cost,
                     SelectionHeuristic heuristicMethod, float epsilon,
                     float tolerance, float* transposedData, PhiSVMModel* curModel);
PhiSVMModel* performOnlineTraining(int nOldPoints, Kernel_params* kp, float cost,
                     SelectionHeuristic heuristicMethod, float epsilon,
                     float tolerance, PhiSVMModel* curModel);

template <int Kernel>
void firstOrder(float* devData, int devDataPitchInFloats,
                float* devTransposedData, int devTransposedDataPitchInFloats,
                float* devLabels, int nPoints, int nDimension, float epsilon,
                float cEpsilon, float* devAlpha, float* devF, float alpha1Diff,
                float alpha2Diff, int iLow, int iHigh, float parameterA,
                float parameterB, float parameterC, float* devCache,
                int devCachePitchInFloats, int iLowCacheIndex,
                int iHighCacheIndex, float* devKernelDiag, void* devResult,
                float cost, bool iLowCompute, bool iHighCompute, int nthreads);
void launchFirstOrder(bool iLowCompute, bool iHighCompute, int kType,
                      int nPoints, int nDimension, float* devData,
                      int devDataPitchInFloats, float* devTransposedData,
                      int devTransposedDataPitchInFloats, float* devLabels,
                      float epsilon, float cEpsilon, float* devAlpha,
                      float* devF, float sAlpha1Diff, float sAlpha2Diff,
                      int iLow, int iHigh, float parameterA, float parameterB,
                      float parameterC, float* devCache,
                      int devCachePitchInFloats, int iLowCacheIndex,
                      int iHighCacheIndex, float* devKernelDiag,
                      void* devResult, float cost, int nthreads);
template <int Kernel>
void secondOrder(float* devData, int devDataPitchInFloats,
                 float* devTransposedData, int devTransposedDataPitchInFloats,
                 float* devLabels, int nPoints, int nDimension, float epsilon,
                 float cEpsilon, float* devAlpha, float* devF, float alpha1Diff,
                 float alpha2Diff, int iLow, int iHigh, float parameterA,
                 float parameterB, float parameterC, float* devCache,
                 int devCachePitchInFloats, int iLowCacheIndex,
                 int iHighCacheIndex, float* devKernelDiag, void* devResult,
                 float cost, float bHigh, bool iHighCompute, int nthreads);
void launchSecondOrder(bool iLowCompute, bool iHighCompute, int kType,
                       int nPoints, int nDimension, float* devData,
                       int devDataPitchInFloats, float* devTransposedData,
                       int devTransposedDataPitchInFloats, float* devLabels,
                       float epsilon, float cEpsilon, float* devAlpha,
                       float* devF, float sAlpha1Diff, float sAlpha2Diff,
                       int iLow, int iHigh, float parameterA, float parameterB,
                       float parameterC, NewCache* kernelCache, float* devCache,
                       int devCachePitchInFloats, int iLowCacheIndex,
                       int iHighCacheIndex, float* devKernelDiag,
                       void* devResult, float cost, int iteration,
                       int nthreads);
template <int Kernel>
void initializeArrays(float* devData, int devDataPitchInFloats, float* devCache,
                      int devCachePitchInFloats, int nPoints, int nDimension,
                      float parameterA, float parameterB, float parameterC,
                      float* devKernelDiag, float* devAlpha, float* devF,
                      float* devLabels, int nthreads);
void launchInitialization(float* devData, int devDataPitchInFloats,
                          float* devCache, int devCachePitchInFloats,
                          int nPoints, int nDimension, int kType,
                          float parameterA, float parameterB, float parameterC,
                          float* devKernelDiag, float* devAlpha, float* devF,
                          float* devLabels, int nthreads);
template <int Kernel>
void takeFirstStep(void* devResult, float* devKernelDiag, float* devData,
                   int devDataPitchInFloats, float* devCache,
                   int devCachePitchInFloats, float* devAlpha, float cost,
                   int nDimension, int iLow, int iHigh, float parameterA,
                   float parameterB, float parameterC);
void launchTakeFirstStep(void* devResult, float* devKernelDiag, float* devData,
                         int devDataPitchInFloats, float* devCache,
                         int devCachePitchInFloats, float* devAlpha, float cost,
                         int nDimension, int iLow, int iHigh, int kType,
                         float parameterA, float parameterB, float parameterC,
                         int nthreads);
void performClassification(float* data, int nData, int nDimension,
                           Kernel_params* kp, float** p_result, PhiSVMModel* model);
void computeKernels(float* devNorms, int devNormsPitchInFloats, float* devAlpha,
                    int nPoints, int nSV, const KernelType kType, int degree,
                    float b, float* devResult);
float kernel(const float v, const int degree, const KernelType kType);
void makeSelfDots(float* devSource, int devSourcePitchInFloats, float* devDest,
                  int sourceCount, int sourceLength);
void makeDots(float* devDots, int devDotsPitchInFloats, float* devSVDots,
              float* devDataDots, int nSV, int nPoints);

class DumpModel {
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & nSamples;
    ar & nDimension;
    for (int i=0; i<nSamples*nDimension; i++) {
      ar & trainingData[i];
    }
    ar & phiSVMModel->nSamples;  // redundant with Model::nSamples
    ar & phiSVMModel->nDimension;  // for precomputed kernel, should be the same as nSamples
    ar & phiSVMModel->epsilon;
    ar & phiSVMModel->bLow;
    ar & phiSVMModel->bHigh;
    for (int i=0; i<phiSVMModel->nSamples; i++) {
      ar & phiSVMModel->alpha[i];
    }
    for (int i=0; i<phiSVMModel->nSamples; i++) {
      ar & phiSVMModel->f[i];
    }
    for (int i=0; i<phiSVMModel->nSamples; i++) {
      ar & phiSVMModel->kernelDiag[i];
    }
    // the data here for precomputed kernel is the kernel matrix
    for (int i=0; i<phiSVMModel->nSamples*phiSVMModel->nDimension; i++) {
      ar & phiSVMModel->data[i];
    }
    for (int i=0; i<phiSVMModel->nSamples; i++) {
      ar & phiSVMModel->labels[i];
    }
  }
public:
  int nSamples;
  int nDimension;
  float* trainingData;
  PhiSVMModel* phiSVMModel;
  DumpModel() {}
  DumpModel(int s, int d) {
    nSamples = s;
    nDimension = d;
    trainingData = new float[s*d];
    phiSVMModel = new PhiSVMModel(MAX_POINTS, MAX_POINTS);
  }
  ~DumpModel() {
    //delete trainingData;
    //delete phiSVMModel->data;
    //delete phiSVMModel->labels;
    //delete phiSVMModel;
  }
};

std::string serialize_DumpModel(DumpModel* model_ptr);
DumpModel* deserialize_DumpModel(std::string, int, int);
void DumpModelToDisk(std::string modelStr);
#endif