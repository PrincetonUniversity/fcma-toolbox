#include "common.h"
#include <iostream>
#include <cmath>
#include <string.h>
#include <cstdlib>
using namespace std;
void GetSAndH(float* data, float* data2, double* beta, int length, int* labels, int rank, double* S, double* H, int v1, int v2, int n);
double* GetInverseMat(double* mat, int rank);
double DoIteration(float* data, float* data2, int length, int* labels, double epsilon, int rank, int v1, int v2);
double DoIteration2(float* dataHead, int offset1, float* dataHead2, int offset2, int length, int* labels, double epsilon, int rank);
