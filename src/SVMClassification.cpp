/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "SVMClassification.h"
#include "LibSVM.h"
#include "common.h"

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
    cerr<<"the master node isn't supposed to do classification jobs"<<endl;
    exit(1);
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
    SVMParameter* param = SetSVMParameter(4); //0 for linear, 4 for precomputed kernel
    (scores+count)->vid = rowBase+i/row;
    (scores+count)->score = DoSVM(nFolds, prob, param);
    //if (me == 0)
    //{
    //  cout<<count<<": "<<(scores+count)->score<<" "<<flush;
    //}
    delete param;
    delete prob->y;
    for (int j=0; j<nTrainings; j++)
    {
      delete prob->x[j];
    }
    delete prob->x;
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
  prob->y = new double[nTrainings];
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
  prob->y = new double[nTrainings];
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
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nTrainings, nTrainings, row, 1.0, corrMatrix, row, corrMatrix, row, 0.0, simMatrix, nTrainings);
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
  delete simMatrix;
  delete corrMatrix;
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
    //cout<<target[i]<<" "<<prob->y[i]<<endl; getchar();
		if(target[i] == prob->y[i])
    {
			total_correct++;
    }
  }
  //cout<<total_correct<<" "<<prob->l; getchar();
  delete target;
  return 1.0*total_correct/prob->l;
}
