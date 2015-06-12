/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"
#include "SVMPredictorWithMasks.h"
#include "MatComputation.h"
#include "Preprocessing.h"
#include "FileProcessing.h"
#include "SVMPredictor.h"
#include <nifti1_io.h>

using namespace std;

/***************************************
Get two parts of the brain to compute the correlation and then use the correlation vectors to predict
input: the raw activation matrix arrays, the number of subjects, the number of subjects, the first mask file, the second mask file, the number of blocks(trials), the blocks, the number of test samples
output: the results are displayed on the screen and returned
****************************************/
int SVMPredictCorrelationWithMasks(RawMatrix** r_matrices1, RawMatrix** r_matrices2, int nSubs, const char* maskFile1, const char* maskFile2, int nTrials, Trial* trials, int nTests, int is_quiet_mode)
{
#ifndef __MIC__
  int i, j;
  svm_set_print_string_function(&print_null);
  RawMatrix** masked_matrices1=NULL;
  RawMatrix** masked_matrices2=NULL;
  if (maskFile1!=NULL)
    masked_matrices1 = GetMaskedMatrices(r_matrices1, nSubs, maskFile1);
  else
    masked_matrices1 = r_matrices1;
  if (maskFile2!=NULL)
    masked_matrices2 = GetMaskedMatrices(r_matrices2, nSubs, maskFile2);
  else
    masked_matrices2 = r_matrices2;
  cout<<"masked matrices generating done!"<<endl;
  cout<<"#voxels for mask1: "<<masked_matrices1[0]->row<<" #voxels for mask2: "<<masked_matrices2[0]->row<<endl;
  float* simMatrix = new float[nTrials*nTrials];
  int corrRow = masked_matrices1[0]->row;
  memset((void*)simMatrix, 0, nTrials*nTrials*sizeof(float));
  int sr = 0, rowLength = 100;
  int result = 0;
  while (sr<corrRow)
  {
    if (rowLength >= corrRow - sr)
    {
      rowLength = corrRow - sr;
    }
    float* tempSimMatrix = GetPartialInnerSimMatrixWithMasks(nSubs, nTrials, sr, rowLength, trials, masked_matrices1, masked_matrices2);
    for (i=0; i<nTrials*nTrials; i++) simMatrix[i] += tempSimMatrix[i];
    delete[] tempSimMatrix;
    sr += rowLength;
  }
  SVMParameter* param = SetSVMParameter(PRECOMPUTED); //LINEAR or PRECOMPUTED
  SVMProblem* prob = GetSVMTrainingSet(simMatrix, nTrials, trials, nTrials-nTests);
  struct svm_model *model = svm_train(prob, param);
  int nTrainings = nTrials-nTests;
  SVMNode* x = new SVMNode[nTrainings+2];
  double predict_distances[nTrials-nTrainings];
  memset(predict_distances, 0, (nTrials-nTrainings)*sizeof(double)); // bds init
  bool predict_correctness[nTrials-nTrainings];
  memset(predict_correctness, false, (nTrials-nTrainings)*sizeof(bool)); // bds init
  for (i=nTrainings; i<nTrials; i++)
  {
    x[0].index = 0;
    x[0].value = i-nTrainings+1;
    for (j=0; j<nTrainings; j++)
    {
      x[j+1].index = j+1;
      x[j+1].value = simMatrix[i*nTrials+j];
    }
    x[j+1].index = -1;
    predict_distances[i-nTrainings] = svm_predict_distance(model, x);
    //int predict_label = predict_distances[j-nTrainings]>0?0:1;
    int predict_label = int(svm_predict(model, x));
    if (trials[i].label == predict_label)
    {
      result++;
      predict_correctness[i-nTrainings] = true;
    }
    else
    {
      predict_correctness[i-nTrainings] = false;
    }
  }
  if (!is_quiet_mode)
  {
    cout<<"blocking testing confidence:"<<endl;
    for (i=nTrainings; i<nTrials; i++)
    {
      cout<<fabs(predict_distances[i-nTrainings])<<" (";
      if (predict_correctness[i-nTrainings])
      {
        cout<<"Correct) ";
      }
      else
      {
        cout<<"Incorrect) ";
      }
    }
    cout<<endl;
  }
  svm_free_and_destroy_model(&model);
  delete[] x;
  delete prob->y;
  for (i=0; i<nTrainings; i++)
  {
    delete prob->x[i];
  }
  delete prob->x;
  delete prob;
  svm_destroy_param(param);
  delete[] simMatrix;
  for (i=0; i<nSubs; i++)
  {
    delete masked_matrices1[i]->matrix;
    if (maskFile2!=NULL) delete masked_matrices2[i]->matrix;
  }
  delete masked_matrices1;
  if (maskFile2!=NULL) delete masked_matrices2;
  return result;
#else
  return 0;
#endif
}

/***********************************************
Get the inner product of vectors from start row(sr), last rowLength-length
input: the number of subjects, the number of blocks, the start row, the number of voxels of masked matrix one that involved in the computing, the trials information, the first masked data array, the second masked data array
output: the partial similarity matrix based on the selected rows of first matrices and the whole second matrices
************************************************/
float* GetPartialInnerSimMatrixWithMasks(int nSubs, int nTrials, int sr, int rowLength, Trial* trials, RawMatrix** masked_matrices1, RawMatrix** masked_matrices2) // compute the correlation between masked matrices
{
  int i;
  int row1 = masked_matrices1[0]->row;
  int row2 = masked_matrices2[0]->row;  //rows should be the same across subjects since we are using the same mask file to filter out voxels
  float* values= new float[nTrials*rowLength*row2];
  float* simMatrix = new float[nTrials*nTrials];
  memset((void*)simMatrix, 0, nTrials*nTrials*sizeof(float));
  for (i=0; i<nTrials; i++)
  {
    int sc = trials[i].sc;
    int ec = trials[i].ec;
    int sid = trials[i].sid;
    int col = masked_matrices1[sid]->col; // the column of 1 and 2 should be the same, i.e. the number of TRs of a block; columns may be different, since different subjects have different TRs
    float* mat1 = masked_matrices1[sid]->matrix;
    float* mat2 = masked_matrices2[sid]->matrix;
    float* buf1 = new float[row1*col]; // col is more than what really need, just in case
    float* buf2 = new float[row2*col]; // col is more than what really need, just in case
    int ml1 = getBuf(sc, ec, row1, col, mat1, buf1);  // get the normalized matrix, return the length of time points to be computed
    int ml2 = getBuf(sc, ec, row2, col, mat2, buf2);  // get the normalized matrix, return the length of time points to be computed, m1==m2
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rowLength, row2, ml1, 1.0, buf1+sr*ml1, ml1, buf2, ml2, 0.0, values+i*rowLength*row2, row2);
    delete[] buf1;
    delete[] buf2;
  }
  NormalizeCorrValues(values, nTrials, rowLength, row2, nSubs);
  GetDotProductUsingMatMul(simMatrix, values, nTrials, rowLength, row2);
  delete[] values;
  return simMatrix;
}


/***************************************
Get one part of the brain to compute the averaged activation and then use the normalized activation vectors to predict
input: the raw activation matrix array, the number of voxels, the number of subjects, the ROI mask file, the number of blocks(trials), the blocks, the number of test samples
output: the results are displayed on the screen and returned
****************************************/
int SVMPredictActivationWithMasks(RawMatrix** avg_matrices, int nSubs, const char* maskFile, int nTrials, Trial* trials, int nTests, int is_quiet_mode)
{
#ifndef __MIC__
  int i, j;
  int nTrainings = nTrials-nTests;
  SVMParameter* param = SetSVMParameter(LINEAR); //LINEAR or PRECOMPUTED
  SVMProblem* prob = new SVMProblem();
  prob->l = nTrainings;
  prob->y = new schar[nTrainings];
  prob->x = new SVMNode*[nTrainings];
  svm_set_print_string_function(&print_null);
  
  RawMatrix** masked_matrices=NULL;
  if (maskFile!=NULL)
    masked_matrices = GetMaskedMatrices(avg_matrices, nSubs, maskFile);
  else
    masked_matrices = avg_matrices;
  cout<<"masked matrices generating done!"<<endl;
  cout<<"#voxels for mask "<<masked_matrices[0]->row<<endl;
  int nVoxels = masked_matrices[0]->row;
  for (i=0; i<nTrainings; i++)
  {
    int sid = trials[i].sid;
    prob->y[i] = trials[i].label;
    prob->x[i] = new SVMNode[nVoxels+1];
    for (j=0; j<nVoxels; j++)
    {
      prob->x[i][j].index = j+1;
      int col = masked_matrices[sid]->col;
      int offset = trials[i].tid_withinsubj;
      prob->x[i][j].value = masked_matrices[sid]->matrix[j*col+offset];
    }
    prob->x[i][j].index = -1;
  }
  struct svm_model *model = svm_train(prob, param);
  SVMNode* x = new SVMNode[nVoxels+1];
  double predict_distances[nTrials-nTrainings];
  memset(predict_distances,0,(nTrials-nTrainings)*sizeof(double)); // bds init
  bool predict_correctness[nTrials-nTrainings];
  memset(predict_correctness, false,(nTrials-nTrainings)*sizeof(bool)); // bds init
  int result = 0;
  for (i=nTrainings; i<nTrials; i++)
  {
    int sid = trials[i].sid;
    for (j=0; j<nVoxels; j++)
    {
      x[j].index = j+1;
      int col = masked_matrices[sid]->col;
      int offset = trials[i].tid_withinsubj;
      x[j].value = masked_matrices[sid]->matrix[j*col+offset];
    }
    x[j].index = -1;
    predict_distances[i-nTrainings] = svm_predict_distance(model, x);
    //int predict_label = predict_distances[j-nTrainings]>0?0:1;
    int predict_label = int(svm_predict(model, x));
    if (trials[i].label == predict_label)
    {
      result++;
      predict_correctness[i-nTrainings] = true;
    }
    else
    {
      predict_correctness[i-nTrainings] = false;
    }
  }
  if (!is_quiet_mode)
  {
    cout<<"blocking testing confidence:"<<endl;
    for (i=nTrainings; i<nTrials; i++)
    {
      cout<<fabs(predict_distances[i-nTrainings])<<" (";
      if (predict_correctness[i-nTrainings])
      {
        cout<<"Correct) ";
      }
      else
      {
        cout<<"Incorrect) ";
      }
    }
    cout<<endl;
  }
  svm_free_and_destroy_model(&model);
  delete[] x;
  delete[] prob->y;
  for (i=0; i<nTrainings; i++)
  {
    delete prob->x[i];
  }
  delete[] prob->x;
  delete prob;
  svm_destroy_param(param);
  for (i=0; i<nSubs; i++)
  {
    delete masked_matrices[i]->matrix;
  }
  delete masked_matrices;
  return result;
#else
  return 0;
#endif
}
