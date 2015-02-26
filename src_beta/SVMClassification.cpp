/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "SVMClassification.h"
#include "MatComputation.h"
//#include "LibSVM.h"
#include "common.h"
#include "ErrorHandling.h"

/****************************************
get the SVM results of classifying correlation vectors of two categories for every voxel
the linear kernel of libSVM is applied here
input: the node id, the correlation matrix array, the number of blocks, the number of folds in the cross validation
output: a list of voxels' scores in terms of SVM accuracies
*****************************************/
VoxelScore* GetSVMPerformance(int me, CorrMatrix** c_matrices, int nTrainings, int nFolds)  //classifiers for a c_matrix array
{
  if (me==0)  //sanity check
  {
    FATAL("the master node isn't supposed to do classification jobs");
  }
  svm_set_print_string_function(&print_null);
  int rowBase = c_matrices[0]->sr;  // assume all elements in c_matrices array have the same starting row
  int row = c_matrices[0]->nVoxels; // assume all elements in c_matrices array have the same #voxels
  int length = row * c_matrices[0]->step; // assume all elements in c_matrices array have the same step, get the number of entries of a coorelation matrix, notice the row here!!
  VoxelScore* scores = new VoxelScore[c_matrices[0]->step];  // get step voxels classification accuracy here
  int i;
  #pragma omp parallel for private(i)
  for (i=0; i<length; i+=row)
  {
    int count = i / row;
    //SVMProblem* prob = GetSVMProblem(c_matrices, row, i, nTrainings);
    SVMProblem* prob = GetSVMProblemWithPreKernel(c_matrices, row, i, nTrainings);
    SVMParameter* param = SetSVMParameter(PRECOMPUTED); //LINEAR or PRECOMPUTED
    (scores+count)->vid = rowBase+i/row;
    (scores+count)->score = DoSVM(nFolds, prob, param);
    //if (me == 0)
    //{
    //  cout<<count<<": "<<(scores+count)->score<<" "<<flush;
    //}
    delete param;
    delete[] prob->y;
    for (int j=0; j<nTrainings; j++)
    {
      delete prob->x[j];
    }
    delete[] prob->x;
    delete prob;
  }
  //if (me == 0)
  //{
  //  cout<<endl;
  //}
  return scores;
}

/*****************************************
generate a SVM classification problem
input: the correlation matrix array, the number of blocks, the number of voxels (actually the length of a correlation vector), the voxel id, the number of training samples
output: the SVM problem described in the libSVM recognizable format
******************************************/
SVMProblem* GetSVMProblem(CorrMatrix** c_matrices, int row, int startIndex, int nTrainings)
{
  SVMProblem* prob = new SVMProblem();
  prob->l = nTrainings;
  prob->y = new schar[nTrainings];
  prob->x = new SVMNode*[nTrainings];
  int i, j;
  for (i=0; i<nTrainings; i++)
  {
    prob->y[i] = c_matrices[i]->tlabel;
    prob->x[i] = new SVMNode[row+1];
    for (j=0; j<row; j++)
    {
      prob->x[i][j].index = j+1;
      prob->x[i][j].value = c_matrices[i]->matrix[startIndex+j];
    }
    prob->x[i][j].index = -1;
  }
  return prob;
}

/*****************************************
generate a SVM classification problem with a precomputed kernel
input: the correlation matrix array, the number of blocks, the number of voxels (actually the length of a correlation vector), the voxel id, the number of training samples
output: the SVM problem described in the libSVM recognizable format
******************************************/
SVMProblem* GetSVMProblemWithPreKernel(CorrMatrix** c_matrices, int row, int startIndex, int nTrainings)
{
  SVMProblem* prob = new SVMProblem();
  prob->l = nTrainings;
  prob->y = new schar[nTrainings];
  prob->x = new SVMNode*[nTrainings];
  int i, j;
  float* simMatrix = new float[nTrainings*nTrainings];
  float* corrMatrix = new float[nTrainings*row];
  for (i=0; i<nTrainings; i++)
  {
    for (j=0; j<row; j++)
    {
      corrMatrix[i*row+j] = c_matrices[i]->matrix[startIndex+j];
    }
    //memcpy(corrMatrix+i*row, (c_matrices[i]->matrix)+startIndex, sizeof(float)*row);
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nTrainings, nTrainings, row, 1.0, corrMatrix, row, corrMatrix, row, 0.0, simMatrix, nTrainings);
  //matmul(corrMatrix, corrMatrix, simMatrix, nTrainings, row, nTrainings);
  for (i=0; i<nTrainings; i++)
  {
    prob->y[i] = c_matrices[i]->tlabel;
    prob->x[i] = new SVMNode[nTrainings+2];
    prob->x[i][0].index = 0;
    prob->x[i][0].value = i+1;
    for (j=0; j<nTrainings; j++)
    {
      prob->x[i][j+1].index = j+1;
      prob->x[i][j+1].value = simMatrix[i*nTrainings+j];
    }
    prob->x[i][j+1].index = -1;
  }
  delete[] simMatrix;
  delete[] corrMatrix;
  return prob;
}

/******************************************
do the SVM cross validation to get the accuracy
input: the number of training samples (will do the cross validation in this training set), the number of folds, the SVM problem, the SVM parameters
output: the accuracy got after the cross validation
*******************************************/
float DoSVM(int nFolds, SVMProblem* prob, SVMParameter* param)
{
  int total_correct = 0;
  int i;
  double* target = new double[prob->l];
  svm_cross_validation_no_shuffle(prob, param, nFolds, target);  // 17 subjects, so do 17-fold
  //svm_cross_validation(prob, param, 8, target);  // 8-fold cross validation
  for(i=0;i<prob->l;i++)
  {
    if(target[i] == prob->y[i])
    {
      total_correct++;
    }
  }
  delete[] target;
  return 1.0*total_correct/prob->l;
}

VoxelScore* GetVoxelwiseSVMPerformance(int me, Trial* trials, Voxel* voxels, int step, int nTrainings, int nFolds)  //classifiers for a voxel array
{
  if (me==0)  //sanity check
  {
    FATAL("the master node isn't supposed to do classification jobs");
  }
  svm_set_print_string_function(&print_null);
  int row = voxels->nVoxels; // assume all elements in voxels array have the same #voxels
  //int length = row * step; // assume all elements in c_matrices array have the same step, get the number of entries of a coorelation matrix, notice the row here!!
  VoxelScore* scores = new VoxelScore[step];  // get step voxels classification accuracy here
  SVMProblem* prob[step];
  SVMParameter* param[step];
  int i;
#if __MEASURE_TIME__
  float t;
  struct timeval start, end;
  gettimeofday(&start, 0);
#endif
  #pragma omp parallel for private(i)
  for (i=0; i<step; i++)
  {
    prob[i] = GetSVMProblemWithPreKernel2(trials, voxels, i, row, nTrainings);
    param[i] = SetSVMParameter(PRECOMPUTED); //LINEAR or PRECOMPUTED
  }
#if __MEASURE_TIME__
  gettimeofday(&end, 0);
  t=end.tv_sec-start.tv_sec+(end.tv_usec-start.tv_usec)*0.000001;
  cout<<"computing time: "<<t<<endl;
  gettimeofday(&start, 0);
#endif
  #pragma omp parallel for
  for (i=0; i<step; i++)
  {
    (scores+i)->vid = voxels->vid[i];
    (scores+i)->score = DoSVM(nFolds, prob[i], param[i]);
    delete param[i];
    delete[] prob[i]->y;
    for (int j=0; j<nTrainings; j++)
    {
      kmp_free(prob[i]->x[j]);
    }
    delete[] prob[i]->x;
    delete prob[i];
  }
#if __MEASURE_TIME__
  gettimeofday(&end, 0);
  t=end.tv_sec-start.tv_sec+(end.tv_usec-start.tv_usec)*0.000001;
  cout<<"svm time: "<<t<<endl;
#endif
  return scores;
}

SVMProblem* GetSVMProblemWithPreKernel2(Trial* trials, Voxel* voxel, int step_id, int row, int nTrainings)  //for voxelwise
{
  SVMProblem* prob = new SVMProblem();
  prob->l = nTrainings;
  prob->y = new schar[nTrainings];
  prob->x = new SVMNode*[nTrainings];
  int i, j;
  float* simMatrix = new float[nTrainings*nTrainings];
  for (i=0; i<nTrainings*nTrainings; i++) simMatrix[i]=0.0f;
  float* corr_vecs = voxel->corr_vecs+step_id*voxel->nTrials*voxel->nVoxels;
  //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nTrainings, nTrainings, row, 1.0, corr_vecs, row, corr_vecs, row, 0.0, simMatrix, nTrainings);
#ifdef __MIC__
  custom_ssyrk((const int)nTrainings, (const int)row, corr_vecs, (const int)row, simMatrix, (const int)nTrainings); // lower triangle matrix
#else
  cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, nTrainings, row, 1.0, corr_vecs, row, 0.0, simMatrix, nTrainings);
#endif

  for (i=0; i<nTrainings; i++)
  {
    prob->y[i] = trials[i].label;
    prob->x[i] = (SVMNode*)kmp_malloc(sizeof(SVMNode)*(nTrainings+2));
    prob->x[i][0].index = 0;
    prob->x[i][0].value = i+1;
    for (j=0; j<nTrainings; j++)
    {
      prob->x[i][j+1].index = j+1;
      prob->x[i][j+1].value = i>=j?simMatrix[i*nTrainings+j]:simMatrix[j*nTrainings+i];
    }
    prob->x[i][j+1].index = -1;
  }
  delete[] simMatrix;
  return prob;
}

VoxelScore* GetVoxelwiseNewSVMPerformance(int me, Trial* trials, Voxel* voxels, int step, int nTrainings, int nFolds)  //classifiers for a voxel array
{
  if (me==0)  //sanity check
  {
    FATAL("the master node isn't supposed to do classification jobs");
  }
  int row = voxels->nVoxels; // assume all elements in voxels array have the same #voxels
  //int length = row * step; // assume all elements in c_matrices array have the same step, get the number of entries of a coorelation matrix, notice the row here!!
  VoxelScore* scores = new VoxelScore[step];  // get step voxels classification accuracy here
  int i;
#ifdef __MEASURE_TIME__
  float t1=0, t2=0;
#endif
  #pragma omp parallel for private(i)
  for (i=0; i<step; i++)
  {
#ifdef __MEASURE_TIME__
    struct timeval start_time, end_time;
    gettimeofday(&start_time, 0);
#endif
    float* data=NULL;
    float* labels=NULL;
    GetNewSVMProblemWithPreKernel(trials, voxels, i, row, nTrainings, &data, &labels);
    (scores+i)->vid = voxels->vid[i];
#ifdef __MEASURE_TIME__
    gettimeofday(&end_time, 0);
    if (omp_get_thread_num()==0)
    {
      t1 += end_time.tv_sec-start_time.tv_sec+(end_time.tv_usec-start_time.tv_usec)*0.000001;
    }
    gettimeofday(&start_time, 0);
#endif
    (scores+i)->score = DOSVMNew(data, nTrainings, nTrainings, nFolds, labels, voxels->vid[i]);
    delete[] data;
    delete[] labels;
#ifdef __MEASURE_TIME__
    gettimeofday(&end_time, 0);
    if (omp_get_thread_num()==0)
    {
      t2 += end_time.tv_sec-start_time.tv_sec+(end_time.tv_usec-start_time.tv_usec)*0.000001;
    }
#endif
  }
#ifdef __MEASURE_TIME__
  if (me==1)  // this time is just for omp_thread_0, is not the wall time of omp
  {
    printf("kernel matrix computing time: %fs\n", t1);
    printf("svm cross validation time: %fs\n", t2);
  }
#endif
  return scores;
}

void GetNewSVMProblemWithPreKernel(Trial* trials, Voxel* voxel, int step_id, int row, int nTrainings, float** p_data, float** p_labels)  //for voxelwise
{
  int i, j;
  float* simMatrix = new float[nTrainings*nTrainings];
  float* labels = new float[nTrainings];
  for (i=0; i<nTrainings*nTrainings; i++) simMatrix[i]=0.0f;
  float* corr_vecs = voxel->corr_vecs+step_id*voxel->nTrials*voxel->nVoxels;
#ifdef __MIC__
  custom_ssyrk((const int)nTrainings, (const int)row, corr_vecs, (const int)row, simMatrix, (const int)nTrainings); // lower triangle matrix
#else
  cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, nTrainings, row, 1.0, corr_vecs, row, 0.0, simMatrix, nTrainings);
#endif
  for (i=0; i<nTrainings; i++)
  {
    for (j=0; j<i; j++)
    {
      simMatrix[j*nTrainings+i] = simMatrix[i*nTrainings+j];
    }
  }
  for (i=0; i<nTrainings*nTrainings; i++)
  {
    simMatrix[i] *= .001f;
  }
  for (i=0; i<nTrainings; i++)
  {
    labels[i] = trials[i].label;
  }  
  *p_data = simMatrix;
  *p_labels = labels;
}

float DOSVMNew(float* data, int nPoints, int nDimension, int nFolds, float* labels, int vid)
{
  Kernel_params kp;
  kp.gamma = 0;
	kp.coef0 = 0;
	kp.degree = 3;
	//kp.b doesn't need to be preset
	kp.kernel_type = "precomputed";
  float cost = 10.0f;
  SelectionHeuristic heuristicMethod = ADAPTIVE;
  float tolerance = 1e-3f;
  float epsilon = 1e-5f;
  float accuracy = crossValidationNoShuffle(data, nPoints, nDimension, nFolds, labels, &kp, cost, heuristicMethod, epsilon, tolerance, NULL); //transposedData is not used here
  return accuracy;
}

void custom_ssyrk(
                 const MKL_INT M, 
                 const MKL_INT K, float *A,
                 const MKL_INT lda, 
                 float *C, const MKL_INT ldc)
{
  int m_max = (M / MBLK) * MBLK;
  int n_max = (M / NBLK) * NBLK;
  int k_max = (K / KBLK) * KBLK;

  // Round ldc to nearest 16
  //const MKL_INT ldcc = ((ldc + 15)/16)*16;
  const int n_row_blks = (M + (MBLK-1)) / MBLK;

  float A_T[MBLK*KBLK]; // 6KB
  float * A_local = (float*)_mm_malloc(M*KBLK*sizeof(float), 64); //204*96*4=76.5KB
  float * C_local = (float*)_mm_malloc(n_row_blks*MBLK*M*sizeof(float), 64);  // 208*204*4=165.75KB

  #pragma simd
  for(int i = 0 ; i < n_row_blks*MBLK*M ; i++)
  {
    C_local[i] = 0.0f;
  }

  for(int k = 0 ; k < K ; k+=KBLK)
  {
    // Load tile into local buffer
    if(k < k_max)
    {
      for(int jj = 0 ; jj < M ; jj++)
      {
        for(int kk = 0 ; kk < KBLK ; kk++)
        {
          A_local[kk + jj*KBLK] = A[(k + kk) + (jj) * lda];
        }
      }
    }
    else  // zero-pad last block
    {
      int k_left = K-k_max;
      for(int jj = 0 ; jj < M ; jj++)
      {
        for(int kk = 0 ; kk < k_left; kk++)
        {
          A_local[kk + jj*KBLK] = A[(k + kk) + (jj) * lda];
        }
      }
      for(int jj = 0 ; jj < M ; jj++)
      {
        for(int kk = k_left ; kk < KBLK ; kk++)
        {
          A_local[kk + jj*KBLK] = 0.0f;
        }
      }
    }
    for(int i = 0 ; i < m_max ; i+=MBLK)
    {
      // Local transpose of block (left matrix)
      for(int kk = 0 ; kk < KBLK ; kk++)
      {
        for(int ii = 0 ; ii < MBLK ; ii++)
        {
          A_T[ii + kk*MBLK] = A_local[kk + (i+ii)*KBLK];  // time consuming
        }
      }
      // Multiply blocks
      for(int j = (i/NBLK)*NBLK ; j < n_max ; j+=NBLK)  // compute the lower triangle
      {
        sgemm_assembly(&(A_T[0]), &(A_local[j*KBLK]), &C_local[i*M + j*MBLK],NULL,NULL,NULL);
      }

      // Fill in remaining rows of C by looping over i,k,j (vectorize over i)
      for(int jj = n_max ; jj < M ; jj++)
      {
        for(int kk = 0 ; kk < KBLK ; kk++)
        {
          #pragma simd
          for(int ii = 0  ; ii < MBLK; ii++)
          {
            C_local[(ii+i*M)+jj*MBLK] += A_T[ii + kk*MBLK] * A_local[kk + jj*KBLK]; // time consuming
          }
        }
      }
    }
    // Fill in bottom right corner of the matrix by looping over i,j,k (no vectorization)
    for(int ii = m_max ; ii < M ; ii++)
    {
      int iii = ii % m_max;
      for(int jj = ii ; jj < M ; jj++)
      {
        for(int kk = 0 ; kk < KBLK ; kk++)
        {
          C_local[(iii)+m_max*M+jj*MBLK] += A_local[kk + ii*KBLK] * A_local[kk + jj*KBLK];
        }
      }
    }
  }

  // Copy lower triangle into output array
  for(int i = 0 ; i < M ; i++)
  {
    int iblk = (i / MBLK)*MBLK;
    int ii = i % MBLK;
    for(int j = i ; j < M ; j++)
    {
      C[i + j*ldc] += C_local[ii+iblk*M + j*MBLK];
    }
  }
  _mm_free(A_local);
  _mm_free(C_local);
}

void sgemm_assembly(float* A, float* B, float* C, float* A_prefetch, float* B_prefetch, float* C_prefetch)
{
#ifdef __MIC__
float mic_zero = 0.0;
float* Z = &mic_zero;
    __asm__ __volatile__("movq %0, %%r8\n\t"
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
                        : : "m"(B), "m"(A), "m"(C), "m"(Z) : "r8","r9","r10","r13","r14","r15","zmm0","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");
//#else
//#pragma message ("KERNEL COMPILATION ERROR in: " __FILE__)
//#error No kernel was compiled, lacking support for current architecture?
#endif
}
