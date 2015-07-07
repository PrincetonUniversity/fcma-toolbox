/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/
#ifndef CUSTOMIZED_MATRIX_MULTIPLY
#define CUSTOMIZED_MATRIX_MULTIPLY

#include "common.h"

// use "CMM_INT" in place of "int" here
// so that type-casting is not done in
// inner loops, which is a hit to performance
// and allows forcing 64-bit (>2GB matrices)
#ifdef CMM_INT_IS_MKL_INT

// force all CMM_INTs to MKL_INT which is
// 32-bit on LP64 but 64-bit if "_ilp64" variants
// of MKL libs are linked (and MKL_ILP64 defined)
typedef MKL_INT CMM_INT;

#elif defined(CMM_INT_IS_LONG)

// force all CMM_INTs to be "long" or 64 bit on
// LP64 systems regardless of MKL_INT size
typedef long CMM_INT;

#else

// default, state before 6/25/2015: just use int
// this will cause failures when number of matrix elements is > INT_MAX
// without typecasting array indices and malloc args up to size_t
typedef int CMM_INT;

#endif

static const CMM_INT MBLK = 16;
static const CMM_INT NBLK = 9;
static const CMM_INT KBLK = 96;
static const CMM_INT BLK = 48;
static const CMM_INT BLK2 = 30;
static const CMM_INT COL = 12;
static const CMM_INT MICTH = 240;

void sgemmTranspose(float* mat1, float* mat2, const CMM_INT M, const CMM_INT N, const CMM_INT K, float* output, const CMM_INT ldc);
void sgemmTransposeMerge(TrialData* td1, TrialData* td2, const CMM_INT M, const CMM_INT N, const CMM_INT K, const CMM_INT nPerSubj, const CMM_INT nSubjs, float* output, const CMM_INT ldc, CMM_INT sr);
void NormalizeBlkData(float* data, const CMM_INT MtBLK, const CMM_INT nPerSubj);
void custom_ssyrk(const CMM_INT M, const CMM_INT K, float *A, const CMM_INT lda, float *C, widelock_t Clock, const CMM_INT ldc);
void custom_ssyrk_old(const CMM_INT M, const CMM_INT K, float *A, const CMM_INT lda, float *C, const CMM_INT ldc);
void sgemm_assembly(float* A, float* B, float* C, float* A_prefetch = NULL, float* B_prefetch = NULL, float* C_prefetch = NULL);
#endif
