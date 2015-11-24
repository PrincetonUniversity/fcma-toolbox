/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <map>
#include <list>
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
#include <climits>
#include <iomanip>
#include <stdlib.h>
// define NDEBUG here to disable assertions
// if/when "release" builds are supported
// #define NDEBUG
#include <cassert>

// nifti1_io.h is from nifticlib http://nifti.nimh.nih.gov
// see README in fcma-toolbox.git/deps/
#include <nifti1_io.h>

// SSE/AVX instrinsics
#if defined(_MSC_VER)
// Microsoft C/C++-compatible compiler */
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
// GCC-compatible compiler (gcc,clang,icc) targeting x86/x86-64
#include <x86intrin.h>
#endif

// OpenMP : gcc >= 4.7 or clang-omp http://clang-omp.github.io
#include <omp.h>

// Aligned memory blocks
#ifdef __INTEL_COMPILER
#include <offload.h>
#else
#include <mm_malloc.h>
#endif
#if defined(__GNUC__)
// gcc supports this syntax
#define ALIGNED(x) __attribute__((aligned(x)))
#else
// visual c++, clang, icc
#define ALIGNED(x) __declspec(align(x))
#endif

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

// Matrix multiplication parameters
#define TINYNUM 1e-4
#define LOGISTICTRHEHOLD 1e-6
#define MAXFILENAMELENGTH 300
#define MAXSUBJS 100
#define MAXTRIALPERSUBJ 64
#define MAXSVMITERATION 20000000

// MPI communication tags
#define COMPUTATIONTAG 1
#define LENGTHTAG 2
#define VOXELCLASSIFIERTAG 3
#define ELAPSETAG 4
#define POSITIONTAG 5
#define SECONDORDERTAG 6

enum Task {
  Corr_Based_SVM = 0,
  Corr_Based_Dis,
  Acti_Based_SVM,
  Corr_Sum,
  Corr_Mask_Classification,
  Corr_Mask_Cross_Validation,
  Acti_Mask_Classification,
  Acti_Mask_Cross_Validation,
  Corr_Visualization,
  Marginal_Screening,
  Error_Type = -1
};

typedef unsigned long long uint64;
typedef unsigned short uint16;

typedef struct raw_matrix_t {
  std::string sname;  // subject name (file name without extension)
  int sid;            // subject id
  int row;
  int col;
  int nx, ny, nz;
  float* matrix;
} RawMatrix;

typedef struct trial_data_t {
  int nTrials;
  int nVoxels;
  int nCols;
  int* trialLengths;  // keep all trials data, subject by subject
  int* scs;
  float* data;  // normalized data for correlation
  trial_data_t(int x, int y) : nTrials(x), nVoxels(y) {}
} TrialData;

typedef struct corr_matrix_t {
  int sid;      // subject id
  int tlabel;   // trial label
  int sr;       // starting row id
  int step;     // row of this matrix
  int nVoxels;  // col of this matrix
  float* matrix;
} CorrMatrix;

// Trial: data structure for the start and end point of a trial
typedef struct trial_t {
  int tid;
  int sid;
  int label;
  int sc, ec;
  // tid_withinsubj: block id within each subject,
  // to be used in averaged matrix of searchlight
  int tid_withinsubj;
} Trial;

typedef struct voxel_t {
  int* vid;  // contains global voxel ids
  float* corr_vecs;
  float* kernel_matrices;  // contains precomputed kernel matrix
  int nTrials;  // row
  int nVoxels;  // col
  // voxel_t(int x, int y, int z) : vid(x), nTrials(y), nVoxels(z) {}
} Voxel;

typedef struct voxel_Score_t {
  int vid;
  float score;
} VoxelScore;

// VoxelXYZ: voxel's 3-d coordinates in mm
typedef struct voxelxyz_t {
  int x, y, z;
} VoxelXYZ;

typedef ALIGNED(64) struct WIDELOCK_T {
  omp_lock_t lock;
} widelock_t;

extern unsigned long long total_count;

#endif
