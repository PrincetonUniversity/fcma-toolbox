/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#ifndef COMMON_H
#define COMMON_H

// head files
#include <vector>
#include <map>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
//#include <omp.h>
#include <iomanip>
#include <nifti1_io.h>

#ifdef __INTEL_COMPILER
#include <offload.h>
#else
#include <mm_malloc.h>
#endif

#if defined(__GNUC__)
// gcc supports this syntax
#define ALIGNED(x) __attribute__ ((aligned(x)))
#else
// visual c++, clang, icc
#define ALIGNED(x) __declspec(align(x))
#endif

using namespace std;

// Matrix multiplication parameters
#define TINYNUM 1e-4
#define LOGISTICTRHEHOLD 1e-6
#define MAXFILENAMELENGTH 300
#define MAXSUBJS 100
#define MAXTRIALPERSUBJ 64

typedef unsigned long long uint64;
typedef unsigned short uint16;

typedef struct raw_matrix_t
{
  string sname;  // subject name (file name without extension)
  int sid;  // subject id
  int row;
  int col;
  int nx, ny, nz;
  float* matrix;
}RawMatrix;

/*typedef struct avg_matrix_t
{
  int sid;  // subject id
  int row;
  int col;
  float* matrix;
}AvgMatrix;*/

typedef struct corr_matrix_t
{
  int sid;  // subject id
  int tlabel;  // trial label
  int sr; // starting row id
  int step; // row of this matrix
  int nVoxels;  // col of this matrix
  float* matrix;
}CorrMatrix;

typedef struct trial_t  //data structure for the start and end point of a trial
{
  int tid;
  int sid;
  int label;
  int sc, ec;
  int tid_withinsubj; // block id within each subject, to be used in averaged matrix of searchlight
}Trial;

typedef struct voxel_Score_t
{
  int vid;
  float score;
}VoxelScore;

typedef struct voxelxyz_t  //voxel's 3-d coordinates
{
  int x, y, z;
}VoxelXYZ;

extern unsigned long long counter;

// MPI communication tags
#define COMPUTATIONTAG 1
#define LENGTHTAG 2
#define VOXELCLASSIFIERTAG 3
#define ELAPSETAG 4
#define POSITIONTAG 5
#define SECONDORDERTAG 6

#endif
