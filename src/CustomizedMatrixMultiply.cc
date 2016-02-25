/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "CustomizedMatrixMultiply.h"
#include "common.h"
#include "ErrorHandling.h"

// note bdsinger 6.25.2015
// use of CMM_INT rather than MKL_INT or int or size_t everywhere in this file
//  - this is to allow matrices greater than 2GB (INT_MAX) in size
//  - assumption is that making loop variables 64-bit is preferable to
// typecasting to size_t
//  - this is not tested and may be heavy handed , feel free to revisit this!
//  - you can change size of CMM_INT via the typedefs in header

// e.g. M=120, N=34470, K=12, mat1 M*K, mat2 N*K
void sgemmTranspose(float* mat1, float* mat2, const CMM_INT M, const CMM_INT N,
                    const CMM_INT K, float* output, const CMM_INT ldc) {
  CMM_INT m_max = (M / BLK2) * BLK2;
  CMM_INT n_max = (N / BLK) * BLK;
  float mat_T[K * BLK];
  float output_local[BLK2 * BLK];
  for (CMM_INT r = 0; r < N; r += BLK) {
    if (r < n_max) {
      // transpose
      for (CMM_INT cc = 0; cc < K; cc++) {
        for (CMM_INT rr = 0; rr < BLK; rr++) {
          mat_T[cc * BLK + rr] = mat2[cc + (r + rr) * K];
        }
      }
      for (CMM_INT m = 0; m < M; m += BLK2) {
        for (CMM_INT i = 0; i < BLK2 * BLK; i++) {
          output_local[i] = 0.0f;
        }
        if (m < m_max) {
          for (CMM_INT i = 0; i < BLK2; i++) {
            for (CMM_INT j = 0; j < BLK; j++) {
              for (CMM_INT k = 0; k < K; k++) {
                output_local[i * BLK + j] +=
                    mat1[(m + i) * K + k] * mat_T[k * BLK + j];
              }
            }
          }
          for (CMM_INT i = 0; i < BLK2; i++) {
#pragma vector nontemporal
            for (CMM_INT j = 0; j < BLK; j++) {
              output[(m + i) * ldc + r + j] = output_local[i * BLK + j];
            }
          }
        }  // if m
        else  // last block
        {
          for (CMM_INT i = 0; i < M - m_max; i++) {
            for (CMM_INT j = 0; j < BLK; j++) {
              for (CMM_INT k = 0; k < K; k++) {
                output_local[i * BLK + j] +=
                    mat1[(m + i) * K + k] * mat_T[k * BLK + j];
              }
            }
          }
          for (CMM_INT i = 0; i < M - m_max; i++) {
#pragma vector nontemporal
            for (CMM_INT j = 0; j < BLK; j++) {
              output[(m + i) * ldc + r + j] = output_local[i * BLK + j];
            }
          }
        }  // else m
      }  // for m
    }  // if r
    else  // last block
    {
      for (CMM_INT cc = 0; cc < K; cc++) {
        for (CMM_INT rr = 0; rr < N - n_max; rr++) {
          mat_T[cc * BLK + rr] = mat2[cc + (r + rr) * K];
        }
      }
      for (CMM_INT m = 0; m < M; m += BLK2) {
        for (CMM_INT i = 0; i < BLK2 * BLK; i++) {
          output_local[i] = 0.0f;
        }
        if (m < m_max) {
          for (CMM_INT i = 0; i < BLK2; i++) {
            for (CMM_INT j = 0; j < N - n_max; j++) {
              for (CMM_INT k = 0; k < K; k++) {
                output_local[i * BLK + j] +=
                    mat1[(m + i) * K + k] * mat_T[k * BLK + j];
              }
            }
          }
          for (CMM_INT i = 0; i < BLK2; i++) {
#pragma vector nontemporal
            for (CMM_INT j = 0; j < N - n_max; j++) {
              output[(m + i) * ldc + r + j] = output_local[i * BLK + j];
            }
          }
        }  // if m
        else  // last block
        {
          for (CMM_INT i = 0; i < M - m_max; i++) {
            for (CMM_INT j = 0; j < N - n_max; j++) {
              for (CMM_INT k = 0; k < K; k++) {
                output_local[i * BLK + j] +=
                    mat1[(m + i) * K + k] * mat_T[k * BLK + j];
              }
            }
          }
          for (CMM_INT i = 0; i < M - m_max; i++) {
#pragma vector nontemporal
            for (CMM_INT j = 0; j < N - n_max; j++) {
              output[(m + i) * ldc + r + j] = output_local[i * BLK + j];
            }
          }
        }  // else m
      }  // for m
    }  // else r
  }  // for r
  return;
}

// merge correlation computing and normalization together
// assume that M is small enough so that we don't need to partition it
void sgemmTransposeMerge(TrialData* td1, TrialData* td2, const CMM_INT M,
                         const CMM_INT N1, const CMM_INT N2, const CMM_INT K,
                         const CMM_INT nPerSubj, const CMM_INT nSubjs,
                         float* output, const CMM_INT ldc, CMM_INT sr) {
  float* mat1 = td1->data + sr * K;
  float* mat2 = td2->data;
  const CMM_INT n_max = (N2 / BLK) * BLK;
  const CMM_INT MtBLK = M * BLK;
  const CMM_INT KtBLK = K * BLK;
  const CMM_INT olsize = MtBLK * nPerSubj;
  const CMM_INT mtsize = KtBLK * nPerSubj;

  // remember, can #define NDEBUG at compile time
  // (or at top of common.h) to disable assertions
  assert(n_max > 0);
  assert(MtBLK > 0);
  assert(KtBLK > 0);
  assert(olsize > 0);
  assert(mtsize > 0);

#pragma omp parallel for collapse(2)  // schedule(dynamic)
  for (CMM_INT s = 0; s < nSubjs; s++) {
    for (CMM_INT n = 0; n < N2; n += BLK) {
      float mat_T[mtsize];
      float output_local[olsize];
      if (n < n_max) {
        // transpose
        for (CMM_INT ss = 0; ss < nPerSubj; ss++) {
          CMM_INT cur_col = td1->scs[s * nPerSubj + ss];
          for (CMM_INT cc = 0; cc < K; cc++) {
            for (CMM_INT rr = 0; rr < BLK; rr++) {
              mat_T[ss * KtBLK + cc * BLK + rr] =
                  mat2[cur_col * N2 + cc + (n + rr) * K];
            }
          }
        }
        for (CMM_INT ss = 0; ss < nPerSubj; ss++) {
          for (CMM_INT i = 0; i < MtBLK; i++) {
            output_local[ss * MtBLK + i] = 0.0f;
          }
          CMM_INT cur_col = td1->scs[s * nPerSubj + ss];
          for (CMM_INT i = 0; i < M; i++) {
            for (CMM_INT j = 0; j < BLK; j++) {
              for (CMM_INT k = 0; k < K; k++) {
                output_local[ss * MtBLK + i * BLK + j] +=
                    mat1[cur_col * N1 + i * K + k] *
                    mat_T[ss * KtBLK + k * BLK + j];
              }
            }
          }
        }  // for ss
        // z-scoring etc.
        NormalizeBlkData(output_local, (CMM_INT)MtBLK, nPerSubj);
        for (CMM_INT ss = 0; ss < nPerSubj; ss++) {
          for (CMM_INT i = 0; i < M; i++) {
#pragma vector nontemporal
            for (CMM_INT j = 0; j < BLK; j++) {
              output[s * nPerSubj * N2 + ss * N2 + i * ldc + n + j] =
                  output_local[ss * MtBLK + i * BLK + j];  // i is vid
            }
          }
        }
      }  // if n
      else {
        // transpose
        for (CMM_INT ss = 0; ss < nPerSubj; ss++) {
          CMM_INT cur_col = td1->scs[s * nPerSubj + ss];
          for (CMM_INT cc = 0; cc < K; cc++) {
            for (CMM_INT rr = 0; rr < N2 - n_max; rr++) {
              mat_T[ss * KtBLK + cc * BLK + rr] =
                  mat2[cur_col * N2 + cc + (n + rr) * K];
            }
          }
        }
        for (CMM_INT ss = 0; ss < nPerSubj; ss++) {
          for (CMM_INT i = 0; i < MtBLK; i++) {
            output_local[ss * MtBLK + i] = 0.0f;
          }
          CMM_INT cur_col = td1->scs[s * nPerSubj + ss];
          for (CMM_INT i = 0; i < M; i++) {
            for (CMM_INT j = 0; j < N2 - n_max; j++) {
              for (CMM_INT k = 0; k < K; k++) {
                output_local[ss * MtBLK + i * BLK + j] +=
                    mat1[cur_col * N1 + i * K + k] *
                    mat_T[ss * KtBLK + k * BLK + j];
              }
            }
          }
        }  // for ss
        // z-scoring etc.
        NormalizeBlkData(output_local, (CMM_INT)MtBLK, nPerSubj);
        for (CMM_INT ss = 0; ss < nPerSubj; ss++) {
          for (CMM_INT i = 0; i < M; i++) {
#pragma vector nontemporal
            for (CMM_INT j = 0; j < N2 - n_max; j++) {
              output[s * nPerSubj * N2 + ss * N2 + i * ldc + n + j] =
                  output_local[ss * MtBLK + i * BLK + j];  // i is vid
            }
          }
        }
      }  // else n
    }    // for n
  }      // for s
  return;
}

// data contains nPerSubj number of M*BLK matrices
// normalization contains two steps
// 1. Fisher-transform each value
// 2. z-score across every entry of M*BLK matrices
// or one can treat data as a nPerSubj-row, M*BLK-column matrix
// z-scoring goes across columns
void NormalizeBlkData(float* data, const CMM_INT MtBLK,
                      const CMM_INT nPerSubj) {
#pragma simd
  for (CMM_INT j = 0; j < MtBLK; j++) {
    float mean = 0.0f;
    float std_dev = 0.0f;
    for (CMM_INT b = 0; b < nPerSubj; b++) {
#ifdef __MIC__
      _mm_prefetch((char*)&(data[b * BLK2 * BLK + j + 16]), _MM_HINT_T0);
#endif
      float num = 1.0f + data[b * MtBLK + j];
      float den = 1.0f - data[b * MtBLK + j];
      num = (num <= 0.0f) ? 1e-4 : num;
      den = (den <= 0.0f) ? 1e-4 : den;
      data[b * MtBLK + j] = 0.5f * logf(num / den);
      mean += data[b * MtBLK + j];
      std_dev += data[b * MtBLK + j] * data[b * MtBLK + j];
    }
    mean = mean / (float)nPerSubj;
    std_dev = std_dev / (float)nPerSubj - mean * mean;
    float inv_std_dev = (std_dev <= 0.0f) ? 0.0f : 1.0f / sqrt(std_dev);
    for (CMM_INT b = 0; b < nPerSubj; b++) {
      data[b * MtBLK + j] = (data[b * MtBLK + j] - mean) * inv_std_dev;
    }
  }
}

void custom_ssyrk(const CMM_INT M, const CMM_INT K, float* A, const CMM_INT lda,
                  float* C, widelock_t Clock, const CMM_INT ldc) {
  CMM_INT m_max = (M / MBLK) * MBLK;
  CMM_INT n_max = (M / NBLK) * NBLK;
  CMM_INT k_max = (K / KBLK) * KBLK;

  // Round ldc to nearest 16
  const CMM_INT n_row_blks = (M + (MBLK - 1)) / MBLK;

  float A_T[MBLK * KBLK];
  float* A_local = (float*)_mm_malloc(M * KBLK * sizeof(float), 64);
  float* C_local =
      (float*)_mm_malloc(n_row_blks * MBLK * M * sizeof(float), 64);

#pragma simd
  for (CMM_INT i = 0; i < n_row_blks * MBLK * M; i++) {
    C_local[i] = 0.0f;
  }

  for (CMM_INT k = 0; k < K; k += KBLK) {
    // Load tile CMM_INTo local buffer
    if (k < k_max) {
      for (CMM_INT jj = 0; jj < M; jj++) {
        for (CMM_INT kk = 0; kk < KBLK; kk++) {
          A_local[kk + jj * KBLK] = A[(k + kk) + (jj) * lda];
        }
      }
    } else  // zero-pad last block
    {
      CMM_INT k_left = K - k_max;
      for (CMM_INT jj = 0; jj < M; jj++) {
        for (CMM_INT kk = 0; kk < k_left; kk++) {
          A_local[kk + jj * KBLK] = A[(k + kk) + (jj) * lda];
        }
      }
      for (CMM_INT jj = 0; jj < M; jj++) {
        for (CMM_INT kk = k_left; kk < KBLK; kk++) {
          A_local[kk + jj * KBLK] = 0.0f;
        }
      }
    }
    for (CMM_INT i = 0; i < m_max; i += MBLK) {
      // Local transpose of block (left matrix)
      for (CMM_INT kk = 0; kk < KBLK; kk++) {
        for (CMM_INT ii = 0; ii < MBLK; ii++) {
          A_T[ii + kk * MBLK] = A_local[kk + (i + ii) * KBLK];
        }
      }
      // Multiply blocks
      for (CMM_INT j = (i / NBLK) * NBLK; j < n_max; j += NBLK) {
        sgemm_assembly(&(A_T[0]), &(A_local[j * KBLK]),
                       &C_local[i * M + j * MBLK], NULL, NULL, NULL);
      }

      // Fill in remaining columns of C by looping over i,k,j (vectorize over i)
      for (CMM_INT jj = n_max; jj < M; jj++) {
        for (CMM_INT kk = 0; kk < KBLK; kk++) {
#pragma simd
          for (CMM_INT ii = 0; ii < MBLK; ii++) {
            C_local[(ii + i * M) + jj * MBLK] +=
                A_T[ii + kk * MBLK] * A_local[kk + jj * KBLK];
          }
        }
      }
    }
    // Fill in bottom right corner of the matrix by looping over i,j,k (no
    // vectorization)
    for (CMM_INT ii = m_max; ii < M; ii++) {
      CMM_INT iii = ii % m_max;
      for (CMM_INT jj = ii; jj < M; jj++) {
        for (CMM_INT kk = 0; kk < KBLK; kk++) {
          C_local[(iii) + m_max * M + jj * MBLK] +=
              A_local[kk + ii * KBLK] * A_local[kk + jj * KBLK];
        }
      }
    }
  }
  omp_set_lock(&(Clock.lock));
  // Copy upper triangle CMM_INTo output array
  for (CMM_INT i = 0; i < M; i++) {
    CMM_INT iblk = (i / MBLK) * MBLK;
    CMM_INT ii = i % MBLK;
    for (CMM_INT j = i; j < M; j++) {
      C[i + j * ldc] += C_local[ii + iblk * M + j * MBLK];
    }
  }
  omp_unset_lock(&(Clock.lock));
  _mm_free(A_local);
  _mm_free(C_local);
}

void custom_ssyrk_old(const CMM_INT M, const CMM_INT K, float* A,
                      const CMM_INT lda, float* C, const CMM_INT ldc) {
  CMM_INT m_max = (M / MBLK) * MBLK;
  CMM_INT n_max = (M / NBLK) * NBLK;
  CMM_INT k_max = (K / KBLK) * KBLK;

  // Round ldc to nearest 16
  const CMM_INT n_row_blks = (M + (MBLK - 1)) / MBLK;

  float A_T[MBLK * KBLK];  // 6KB
  float* A_local =
      (float*)_mm_malloc(M * KBLK * sizeof(float), 64);  // 204*96*4=76.5KB
  float* C_local = (float*)_mm_malloc(n_row_blks * MBLK * M * sizeof(float),
                                      64);  // 208*204*4=165.75KB

#pragma simd
  for (CMM_INT i = 0; i < n_row_blks * MBLK * M; i++) {
    C_local[i] = 0.0f;
  }

  for (CMM_INT k = 0; k < K; k += KBLK) {
    // Load tile CMM_INTo local buffer
    if (k < k_max) {
      for (CMM_INT jj = 0; jj < M; jj++) {
        for (CMM_INT kk = 0; kk < KBLK; kk++) {
          A_local[kk + jj * KBLK] = A[(k + kk) + (jj) * lda];
        }
      }
    } else  // zero-pad last block
    {
      CMM_INT k_left = K - k_max;
      for (CMM_INT jj = 0; jj < M; jj++) {
        for (CMM_INT kk = 0; kk < k_left; kk++) {
          A_local[kk + jj * KBLK] = A[(k + kk) + (jj) * lda];
        }
      }
      for (CMM_INT jj = 0; jj < M; jj++) {
        for (CMM_INT kk = k_left; kk < KBLK; kk++) {
          A_local[kk + jj * KBLK] = 0.0f;
        }
      }
    }
    for (CMM_INT i = 0; i < m_max; i += MBLK) {
      // Local transpose of block (left matrix)
      for (CMM_INT kk = 0; kk < KBLK; kk++) {
        for (CMM_INT ii = 0; ii < MBLK; ii++) {
          A_T[ii + kk * MBLK] =
              A_local[kk + (i + ii) * KBLK];  // time consuming
        }
      }
      // Multiply blocks
      for (CMM_INT j = (i / NBLK) * NBLK; j < n_max;
           j += NBLK)  // compute the lower triangle
      {
        sgemm_assembly(&(A_T[0]), &(A_local[j * KBLK]),
                       &C_local[i * M + j * MBLK], NULL, NULL, NULL);
      }

      // Fill in remaining rows of C by looping over i,k,j (vectorize over i)
      for (CMM_INT jj = n_max; jj < M; jj++) {
        for (CMM_INT kk = 0; kk < KBLK; kk++) {
#pragma simd
          for (CMM_INT ii = 0; ii < MBLK; ii++) {
            C_local[(ii + i * M) + jj * MBLK] +=
                A_T[ii + kk * MBLK] *
                A_local[kk + jj * KBLK];  // time consuming
          }
        }
      }
    }
    // Fill in bottom right corner of the matrix by looping over i,j,k (no
    // vectorization)
    for (CMM_INT ii = m_max; ii < M; ii++) {
      CMM_INT iii = ii % m_max;
      for (CMM_INT jj = ii; jj < M; jj++) {
        for (CMM_INT kk = 0; kk < KBLK; kk++) {
          C_local[(iii) + m_max * M + jj * MBLK] +=
              A_local[kk + ii * KBLK] * A_local[kk + jj * KBLK];
        }
      }
    }
  }

  // Copy lower triangle CMM_INTo output array
  for (CMM_INT i = 0; i < M; i++) {
    CMM_INT iblk = (i / MBLK) * MBLK;
    CMM_INT ii = i % MBLK;
    for (CMM_INT j = i; j < M; j++) {
      C[i + j * ldc] += C_local[ii + iblk * M + j * MBLK];
    }
  }
  _mm_free(A_local);
  _mm_free(C_local);
}

void sgemm_assembly(float* A, float* B, float* C, float* A_prefetch,
                    float* B_prefetch, float* C_prefetch) {
#ifdef __MIC__
  float mic_zero = 0.0;
  float* Z = &mic_zero;
  __asm__ __volatile__(
      "movq %0, %%r8\n\t"
      "movq %1, %%r9\n\t"
      "movq %2, %%r10\n\t"
      "movq %3, %%r15\n\t"
      "movq $0, %%r14\n\t"
      "movq $0, %%r13\n\t"
      "10016:\n\t"
      "addq $16, %%r14\n\t"
      "vmovaps (%%r10), %%zmm23\n\t"
      "vprefetch1 64(%%r10)\n\t"
      "vmovaps 64(%%r10), %%zmm24\n\t"
      "vprefetch1 128(%%r10)\n\t"
      "vmovaps 128(%%r10), %%zmm25\n\t"
      "vprefetch1 192(%%r10)\n\t"
      "vmovaps 192(%%r10), %%zmm26\n\t"
      "vprefetch1 256(%%r10)\n\t"
      "vmovaps 256(%%r10), %%zmm27\n\t"
      "vprefetch1 320(%%r10)\n\t"
      "vmovaps 320(%%r10), %%zmm28\n\t"
      "vprefetch1 384(%%r10)\n\t"
      "vmovaps 384(%%r10), %%zmm29\n\t"
      "vprefetch1 448(%%r10)\n\t"
      "vmovaps 448(%%r10), %%zmm30\n\t"
      "vprefetch1 512(%%r10)\n\t"
      "vmovaps 512(%%r10), %%zmm31\n\t"
      "vprefetch1 576(%%r10)\n\t"
      "movq $0, %%r13\n\t"
      "216:\n\t"
      "addq $8, %%r13\n\t"
      "vmovaps 0(%%r9), %%zmm0\n\t"
      "vprefetch0 64(%%r9)\n\t"
      "vfmadd231ps 0(%%r8){{1to16}}, %%zmm0, %%zmm23\n\t"
      "vprefetch0 128(%%r9)\n\t"
      "vfmadd231ps 384(%%r8){{1to16}}, %%zmm0, %%zmm24\n\t"
      "vprefetch0 192(%%r9)\n\t"
      "vfmadd231ps 768(%%r8){{1to16}}, %%zmm0, %%zmm25\n\t"
      "vprefetch0 256(%%r9)\n\t"
      "vfmadd231ps 1152(%%r8){{1to16}}, %%zmm0, %%zmm26\n\t"
      "vfmadd231ps 1536(%%r8){{1to16}}, %%zmm0, %%zmm27\n\t"
      "vfmadd231ps 1920(%%r8){{1to16}}, %%zmm0, %%zmm28\n\t"
      "vfmadd231ps 2304(%%r8){{1to16}}, %%zmm0, %%zmm29\n\t"
      "vfmadd231ps 2688(%%r8){{1to16}}, %%zmm0, %%zmm30\n\t"
      "vfmadd231ps 3072(%%r8){{1to16}}, %%zmm0, %%zmm31\n\t"
      "vmovaps 64(%%r9), %%zmm0\n\t"
      "vprefetch0 320(%%r9)\n\t"
      "vfmadd231ps 4(%%r8){{1to16}}, %%zmm0, %%zmm23\n\t"
      "vprefetch0 384(%%r9)\n\t"
      "vfmadd231ps 388(%%r8){{1to16}}, %%zmm0, %%zmm24\n\t"
      "vprefetch0 448(%%r9)\n\t"
      "vfmadd231ps 772(%%r8){{1to16}}, %%zmm0, %%zmm25\n\t"
      "vprefetch0 512(%%r9)\n\t"
      "vfmadd231ps 1156(%%r8){{1to16}}, %%zmm0, %%zmm26\n\t"
      "vfmadd231ps 1540(%%r8){{1to16}}, %%zmm0, %%zmm27\n\t"
      "vfmadd231ps 1924(%%r8){{1to16}}, %%zmm0, %%zmm28\n\t"
      "vfmadd231ps 2308(%%r8){{1to16}}, %%zmm0, %%zmm29\n\t"
      "vfmadd231ps 2692(%%r8){{1to16}}, %%zmm0, %%zmm30\n\t"
      "vfmadd231ps 3076(%%r8){{1to16}}, %%zmm0, %%zmm31\n\t"
      "vmovaps 128(%%r9), %%zmm0\n\t"
      "vprefetch1 64(%%r9)\n\t"
      "vfmadd231ps 8(%%r8){{1to16}}, %%zmm0, %%zmm23\n\t"
      "vprefetch1 128(%%r9)\n\t"
      "vfmadd231ps 392(%%r8){{1to16}}, %%zmm0, %%zmm24\n\t"
      "vprefetch1 192(%%r9)\n\t"
      "vfmadd231ps 776(%%r8){{1to16}}, %%zmm0, %%zmm25\n\t"
      "vprefetch1 256(%%r9)\n\t"
      "vfmadd231ps 1160(%%r8){{1to16}}, %%zmm0, %%zmm26\n\t"
      "vfmadd231ps 1544(%%r8){{1to16}}, %%zmm0, %%zmm27\n\t"
      "vfmadd231ps 1928(%%r8){{1to16}}, %%zmm0, %%zmm28\n\t"
      "vfmadd231ps 2312(%%r8){{1to16}}, %%zmm0, %%zmm29\n\t"
      "vfmadd231ps 2696(%%r8){{1to16}}, %%zmm0, %%zmm30\n\t"
      "vfmadd231ps 3080(%%r8){{1to16}}, %%zmm0, %%zmm31\n\t"
      "vmovaps 192(%%r9), %%zmm0\n\t"
      "vprefetch1 320(%%r9)\n\t"
      "vfmadd231ps 12(%%r8){{1to16}}, %%zmm0, %%zmm23\n\t"
      "vprefetch1 384(%%r9)\n\t"
      "vfmadd231ps 396(%%r8){{1to16}}, %%zmm0, %%zmm24\n\t"
      "vprefetch1 448(%%r9)\n\t"
      "vfmadd231ps 780(%%r8){{1to16}}, %%zmm0, %%zmm25\n\t"
      "vprefetch1 512(%%r9)\n\t"
      "vfmadd231ps 1164(%%r8){{1to16}}, %%zmm0, %%zmm26\n\t"
      "vfmadd231ps 1548(%%r8){{1to16}}, %%zmm0, %%zmm27\n\t"
      "vfmadd231ps 1932(%%r8){{1to16}}, %%zmm0, %%zmm28\n\t"
      "vfmadd231ps 2316(%%r8){{1to16}}, %%zmm0, %%zmm29\n\t"
      "vfmadd231ps 2700(%%r8){{1to16}}, %%zmm0, %%zmm30\n\t"
      "vfmadd231ps 3084(%%r8){{1to16}}, %%zmm0, %%zmm31\n\t"
      "vmovaps 256(%%r9), %%zmm0\n\t"
      "vprefetch0 64(%%r8)\n\t"
      "vfmadd231ps 16(%%r8){{1to16}}, %%zmm0, %%zmm23\n\t"
      "vprefetch0 448(%%r8)\n\t"
      "vfmadd231ps 400(%%r8){{1to16}}, %%zmm0, %%zmm24\n\t"
      "vprefetch0 832(%%r8)\n\t"
      "vfmadd231ps 784(%%r8){{1to16}}, %%zmm0, %%zmm25\n\t"
      "vprefetch0 1216(%%r8)\n\t"
      "vfmadd231ps 1168(%%r8){{1to16}}, %%zmm0, %%zmm26\n\t"
      "vfmadd231ps 1552(%%r8){{1to16}}, %%zmm0, %%zmm27\n\t"
      "vfmadd231ps 1936(%%r8){{1to16}}, %%zmm0, %%zmm28\n\t"
      "vfmadd231ps 2320(%%r8){{1to16}}, %%zmm0, %%zmm29\n\t"
      "vfmadd231ps 2704(%%r8){{1to16}}, %%zmm0, %%zmm30\n\t"
      "vfmadd231ps 3088(%%r8){{1to16}}, %%zmm0, %%zmm31\n\t"
      "vmovaps 320(%%r9), %%zmm0\n\t"
      "vprefetch0 1600(%%r8)\n\t"
      "vfmadd231ps 20(%%r8){{1to16}}, %%zmm0, %%zmm23\n\t"
      "vprefetch0 1984(%%r8)\n\t"
      "vfmadd231ps 404(%%r8){{1to16}}, %%zmm0, %%zmm24\n\t"
      "vprefetch0 2368(%%r8)\n\t"
      "vfmadd231ps 788(%%r8){{1to16}}, %%zmm0, %%zmm25\n\t"
      "vprefetch0 2752(%%r8)\n\t"
      "vfmadd231ps 1172(%%r8){{1to16}}, %%zmm0, %%zmm26\n\t"
      "vfmadd231ps 1556(%%r8){{1to16}}, %%zmm0, %%zmm27\n\t"
      "vfmadd231ps 1940(%%r8){{1to16}}, %%zmm0, %%zmm28\n\t"
      "vfmadd231ps 2324(%%r8){{1to16}}, %%zmm0, %%zmm29\n\t"
      "vfmadd231ps 2708(%%r8){{1to16}}, %%zmm0, %%zmm30\n\t"
      "vfmadd231ps 3092(%%r8){{1to16}}, %%zmm0, %%zmm31\n\t"
      "vmovaps 384(%%r9), %%zmm0\n\t"
      "vprefetch0 3136(%%r8)\n\t"
      "vfmadd231ps 24(%%r8){{1to16}}, %%zmm0, %%zmm23\n\t"
      "vfmadd231ps 408(%%r8){{1to16}}, %%zmm0, %%zmm24\n\t"
      "vfmadd231ps 792(%%r8){{1to16}}, %%zmm0, %%zmm25\n\t"
      "vfmadd231ps 1176(%%r8){{1to16}}, %%zmm0, %%zmm26\n\t"
      "vfmadd231ps 1560(%%r8){{1to16}}, %%zmm0, %%zmm27\n\t"
      "vfmadd231ps 1944(%%r8){{1to16}}, %%zmm0, %%zmm28\n\t"
      "vfmadd231ps 2328(%%r8){{1to16}}, %%zmm0, %%zmm29\n\t"
      "vfmadd231ps 2712(%%r8){{1to16}}, %%zmm0, %%zmm30\n\t"
      "vfmadd231ps 3096(%%r8){{1to16}}, %%zmm0, %%zmm31\n\t"
      "vmovaps 448(%%r9), %%zmm0\n\t"
      "vfmadd231ps 28(%%r8){{1to16}}, %%zmm0, %%zmm23\n\t"
      "vfmadd231ps 412(%%r8){{1to16}}, %%zmm0, %%zmm24\n\t"
      "vfmadd231ps 796(%%r8){{1to16}}, %%zmm0, %%zmm25\n\t"
      "vfmadd231ps 1180(%%r8){{1to16}}, %%zmm0, %%zmm26\n\t"
      "vfmadd231ps 1564(%%r8){{1to16}}, %%zmm0, %%zmm27\n\t"
      "vfmadd231ps 1948(%%r8){{1to16}}, %%zmm0, %%zmm28\n\t"
      "vfmadd231ps 2332(%%r8){{1to16}}, %%zmm0, %%zmm29\n\t"
      "vfmadd231ps 2716(%%r8){{1to16}}, %%zmm0, %%zmm30\n\t"
      "vfmadd231ps 3100(%%r8){{1to16}}, %%zmm0, %%zmm31\n\t"
      "addq $32, %%r8\n\t"
      "addq $512, %%r9\n\t"
      "cmpq $96, %%r13\n\t"
      "jl 216b\n\t"
      "subq $384, %%r8\n\t"
      "vmovaps %%zmm23, (%%r10)\n\t"
      "vmovaps %%zmm24, 64(%%r10)\n\t"
      "vmovaps %%zmm25, 128(%%r10)\n\t"
      "vmovaps %%zmm26, 192(%%r10)\n\t"
      "vmovaps %%zmm27, 256(%%r10)\n\t"
      "vmovaps %%zmm28, 320(%%r10)\n\t"
      "vmovaps %%zmm29, 384(%%r10)\n\t"
      "vmovaps %%zmm30, 448(%%r10)\n\t"
      "vmovaps %%zmm31, 512(%%r10)\n\t"
      "addq $64, %%r10\n\t"
      "subq $6080, %%r9\n\t"
      "cmpq $16, %%r14\n\t"
      "jl 10016b\n\t"
      :
      : "m"(B), "m"(A), "m"(C), "m"(Z)
      : "r8", "r9", "r10", "r13", "r14", "r15", "zmm0", "zmm23", "zmm24",
        "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31");
#endif
}
