/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/
#ifndef CUSTOMIZED_MATRIX_MULTIPLY
#define CUSTOMIZED_MATRIX_MULTIPLY

#include "common.h"
#include <mkl.h>
#define MBLK  16
#define NBLK  9
#define KBLK  96
#define BLK   48
#define BLK2  60
#define COL   12
#define MICTH 240

void sgemmTranspose(float* mat1, float* mat2, const MKL_INT M, const MKL_INT N, const MKL_INT K, float* output, const MKL_INT ldc);
void sgemmTransposeMerge(TrialData* td1, TrialData* td2, const MKL_INT M, const MKL_INT N, const MKL_INT K, const int nPerSubj, const int nSubjs, float* output, const MKL_INT ldc, int sr);
void NormalizeBlkData(float* data, const int M, const int nPerSubj);
void custom_ssyrk(const MKL_INT M, const MKL_INT K, float *A, const MKL_INT lda, float *C, widelock_t Clock, const MKL_INT ldc);
void custom_ssyrk_old(const MKL_INT M, const MKL_INT K, float *A, const MKL_INT lda, float *C, const MKL_INT ldc);
void sgemm_assembly(float* A, float* B, float* C, float* A_prefetch = NULL, float* B_prefetch = NULL, float* C_prefetch = NULL);
#endif
