/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "SVMClassification.h"
#include "MatComputation.h"
#include "CustomizedMatrixMultiply.h"
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
  //cout<<"computing time: "<<t<<endl;
  gettimeofday(&start, 0);
#endif
  #pragma omp parallel for schedule(dynamic)
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
  //cout<<"svm time: "<<t<<endl;
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
  //float* simMatrix = new float[nTrainings*nTrainings];
  //for (i=0; i<nTrainings*nTrainings; i++) simMatrix[i]=0.0f;
  float* simMatrix = voxel->corr_vecs+step_id*nTrainings*nTrainings;
/*
#ifdef __MIC__
  custom_ssyrk_old((const int)nTrainings, (const int)row, corr_vecs, (const int)row, simMatrix, (const int)nTrainings); // lower triangle matrix
#else
  cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, nTrainings, row, 1.0, corr_vecs, row, 0.0, simMatrix, nTrainings);
#endif
*/
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
  //delete[] simMatrix;
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
  #pragma omp parallel for private(i) schedule(dynamic)
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
    //cout<<omp_get_thread_num()<<" here"<<endl;
    //delete[] data;
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
    //printf("kernel matrix computing time: %fs\n", t1);
    //printf("svm cross validation time: %fs\n", t2);
  }
#endif
  return scores;
}

void GetNewSVMProblemWithPreKernel(Trial* trials, Voxel* voxel, int step_id, int row, int nTrainings, float** p_data, float** p_labels)  //for voxelwise
{
  int i, j;
  //float* simMatrix = new float[nTrainings*nTrainings];
  float* labels = new float[nTrainings];
/*
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
*/
  float* simMatrix = voxel->corr_vecs+step_id*nTrainings*nTrainings;
  for (i=0; i<nTrainings*nTrainings; i++)
  {
    simMatrix[i] *= .001f;
  }
  for (i=0; i<nTrainings; i++)
  {
    for (j=0; j<i; j++)
    {
      simMatrix[j*nTrainings+i] = simMatrix[i*nTrainings+j];
    }
  }
  //if (step_id==1)
  //cout<<simMatrix[0]<<" "<<simMatrix[1]<<endl;sleep(1);exit(1);
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
