#include "common.h"
#include <iostream>
#include <cmath>
#include <string.h>
#include <cstdlib>
#ifdef USE_MKL
#include <mkl.h>
#elif defined __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern "C" {
#include <cblas.h>
}
#endif
using namespace std;
void GetSAndH(float* data, float* data2, double* beta, int length, int* labels, int rank, float* S, float* H, int v1, int v2, int n);
float* GetInverseMat(float* mat, int rank);
float DoIteration(float* data, float* data2, int length, int* labels, float epsilon, int rank, int v1, int v2);
float DoIteration2(float* dataHead, int offset1, float* dataHead2, int offset2, int length, int* labels, float epsilon, int rank);
