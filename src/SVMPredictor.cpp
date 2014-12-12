/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"
#include "SVMPredictor.h"
#include "MatComputation.h"
#include "LibSVM.h"
#include "Preprocessing.h"
#include "FileProcessing.h"
#include "ErrorHandling.h"
//#include "Searchlight.h"

int getNumTopIndices(int* tops, int maxtops, int nvoxels)
{
    int ntops = 0;
    
    if ( (maxtops < 1) || (nvoxels < tops[0]) )
        return 0;
    
    while ( (ntops < maxtops) && (nvoxels >= tops[ntops]) )
        ntops++;
    
    return ntops;
}

/***************************************
predict a new sample based on a trained SVM model and a variation of the numbers of top voxels. if correlation, assume that it's a self correlation, so only one mask file is enough
input: the raw activation matrix array, the average activation matrix array, the number of subjects, the number of blocks(trials), the blocks, the number of test samples, the task type, the files to store the results, the mask file
output: the results are displayed on the screen
****************************************/
void SVMPredict(RawMatrix** r_matrices, RawMatrix** avg_matrices, int nSubs, int nTrials, Trial* trials, int nTests, int taskType, const char* topVoxelFile, const char* mask_file, int is_quiet_mode)
{
#ifndef __MIC__
  RawMatrix** masked_matrices=NULL;
  int row = 0;
  int col = 0;
  svm_set_print_string_function(&print_null);
  VoxelScore* scores = NULL;
  int tops[] = {10, 20, 50, 100, 200, 500, 1000, 2000, 5000};//, 10000, 20000, 40000};
  int maxtops = sizeof(tops)/sizeof((tops)[0]);
  int ntops;
  switch (taskType)
  {
    case 0:
    case 1:
      if (mask_file!=NULL)
        masked_matrices = GetMaskedMatrices(r_matrices, nSubs, mask_file);
      else
        masked_matrices = r_matrices;
      row = masked_matrices[0]->row;
      col = masked_matrices[0]->col;
      scores = ReadTopVoxelFile(topVoxelFile, row);
      RearrangeMatrix(masked_matrices, scores, row, col, nSubs);
      ntops = getNumTopIndices(tops, maxtops, row);
      if (ntops > 0)
          CorrelationBasedClassification(tops, ntops, nSubs, nTrials, trials, nTests, masked_matrices, is_quiet_mode);
      else
          cerr<<"less than "<<tops[0]<<"voxels!"<<endl;
      break;
    case 2:
      if (mask_file!=NULL)
        masked_matrices = GetMaskedMatrices(avg_matrices, nSubs, mask_file);
      else
        masked_matrices = avg_matrices;
      row = masked_matrices[0]->row;
      col = masked_matrices[0]->col;
      scores = ReadTopVoxelFile(topVoxelFile, row);
      RearrangeMatrix(masked_matrices, scores, row, col, nSubs);
      ntops = getNumTopIndices(tops, maxtops, row);
      if (ntops > 0)
          ActivationBasedClassification(tops, ntops, nTrials, trials, nTests, masked_matrices, is_quiet_mode);
      else
          cerr<<"less than "<<tops[0]<<"voxels!"<<endl;
      break;
    default:
      FATAL("Unknown task type");
  }
  delete[] scores;
#endif
}

/**********************************************
do the prediction based on correlation and a variation of the numbers of top voxels
input: the array of the numbers of top voxels, the number of subjects, the number of blocks, the blocks, the number of test samples, the raw activation matrix array, quiet mode
output: the results are displayed on the screen
***********************************************/
void CorrelationBasedClassification(int* tops, int ntops, int nSubs, int nTrials, Trial* trials, int nTests, RawMatrix** r_matrices, int is_quiet_mode)
{
  int i, j, k;
  int col = r_matrices[0]->col;
  float* simMatrix = new float[nTrials*nTrials];
  for (i=0; i<ntops; i++)
  {
    //simMatrix = GetInnerSimMatrix(tops[i], col, nSubs, nTrials, trials, r_matrices);
    for (j=0; j<nTrials*nTrials; j++) simMatrix[j] = 0.0;
    int sr = 0, rowLength = 500;
    while (sr<tops[i])
    {
      if (rowLength >= tops[i] - sr)
      {
        rowLength = tops[i] - sr;
      }
      float* tempSimMatrix = GetPartialInnerSimMatrix(tops[i], col, nSubs, nTrials, sr, rowLength, trials, r_matrices);
      for (j=0; j<nTrials*nTrials; j++) simMatrix[j] += tempSimMatrix[j];
      //cout<<i<<" "<<sr<<" "<<tempSimMatrix[0]<<" "<<tempSimMatrix[1]<<endl;
      delete[] tempSimMatrix;
      sr += rowLength;
    }
    SVMParameter* param = SetSVMParameter(4); // precomputed
    SVMProblem* prob = GetSVMTrainingSet(simMatrix, nTrials, trials, nTrials-nTests);
    struct svm_model *model = svm_train(prob, param);
    int nTrainings = nTrials-nTests;
    SVMNode* x = new SVMNode[nTrainings+2];
    int result = 0;
    double predict_distances[nTrials-nTrainings];
    bool predict_correctness[nTrials-nTrainings];
    for (j=nTrainings; j<nTrials; j++)
    {
      x[0].index = 0;
      x[0].value = j-nTrainings+1;
      for (k=0; k<nTrainings; k++)
      {
        x[k+1].index = k+1;
        x[k+1].value = simMatrix[j*nTrials+k];
      }
      x[k+1].index = -1;
      if (!is_quiet_mode)
        predict_distances[j-nTrainings] = svm_predict_distance(model, x);
      //int predict_label = predict_distances[j-nTrainings]>0?0:1;
      int predict_label = int(svm_predict(model, x));
      if (trials[j].label == predict_label)
      {
        result++;
        predict_correctness[j-nTrainings] = true;
      }
      else
      {
        predict_correctness[j-nTrainings] = false;
      }
    }
    cout<<tops[i]<<": "<<result<<"/"<<nTrials-nTrainings<<"="<<result*1.0/(nTrials-nTrainings)<<endl;
    if (!is_quiet_mode)
    {
      cout<<"blocking testing confidence:"<<endl;
      for (j=nTrainings; j<nTrials; j++)
      {
        cout<<fabs(predict_distances[j-nTrainings])<<" (";
        if (predict_correctness[j-nTrainings])
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
    for (j=0; j<nTrainings; j++)
    {
      delete prob->x[j];
    }
    delete[] prob->x;
    delete prob;
    svm_destroy_param(param);
  }
  delete [] simMatrix; // bds []
}

/**********************************************
do the prediction based on activation and a variation of the numbers of top voxels
input: the array of the numbers of top voxels, the number of blocks, the blocks, the number of test samples, and the raw activation matrix array, quiet mode
output: the results are displayed on the screen
***********************************************/
void ActivationBasedClassification(int* tops, int ntops, int nTrials, Trial* trials, int nTests, RawMatrix** avg_matrices, int is_quiet_mode)
{
  int i, j, k;
  int nTrainings = nTrials-nTests;
  SVMParameter* param = SetSVMParameter(0); // linear
  SVMProblem* prob = new SVMProblem();
  prob->l = nTrainings;
  prob->y = new double[nTrainings];
  prob->x = new SVMNode*[nTrainings];
  for (i=0; i<ntops; i++)
  {
    for (j=0; j<nTrainings; j++)
    {
      int sid = trials[j].sid;
      prob->y[j] = trials[j].label;
      prob->x[j] = new SVMNode[tops[i]+1];
      for (k=0; k<tops[i]; k++)
      {
        prob->x[j][k].index = k+1;
        int col = avg_matrices[sid]->col;
        int offset = trials[j].tid_withinsubj;
        prob->x[j][k].value = avg_matrices[sid]->matrix[k*col+offset];
      }
      prob->x[j][k].index = -1;
    }
    struct svm_model *model = svm_train(prob, param);
    SVMNode* x = new SVMNode[tops[i]+1];
    int result = 0;
    double predict_distances[nTrials-nTrainings];
    memset(predict_distances, 0, (nTrials-nTrainings)*sizeof(double));
    bool predict_correctness[nTrials-nTrainings];
    memset(predict_correctness, false, (nTrials-nTrainings)*sizeof(bool));
    for (j=nTrainings; j<nTrials; j++)
    {
      int sid = trials[j].sid;
      for (k=0; k<tops[i]; k++)
      {
        x[k].index = k+1;
        int col = avg_matrices[sid]->col;
        int offset = trials[j].tid_withinsubj;
        x[k].value = avg_matrices[sid]->matrix[k*col+offset];
      }
      x[k].index = -1;
      predict_distances[j-nTrainings] = svm_predict_distance(model, x);
      //int predict_label = predict_distances[j-nTrainings]>0?0:1;
      int predict_label = int(svm_predict(model, x));
      if (trials[j].label == predict_label)
      {
        result++;
        predict_correctness[j-nTrainings] = true;
      }
      else
      {
        predict_correctness[j-nTrainings] = false;
      }
    }
    cout<<tops[i]<<": "<<result<<"/"<<nTrials-nTrainings<<"="<<result*1.0/(nTrials-nTrainings)<<endl;
    if (!is_quiet_mode)
    {
      cout<<"blocking testing confidence:"<<endl;
      for (j=nTrainings; j<nTrials; j++)
      {
        cout<<fabs(predict_distances[j-nTrainings])<<" (";
        if (predict_correctness[j-nTrainings])
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
  }
  delete [] prob->y;
  for (j=0; j<nTrainings; j++)
  {
    delete prob->x[j];
  }
  delete [] prob->x;
  delete prob;
  svm_destroy_param(param);
}

/*****************************
Read top voxel information
input: top voxel file, the number of voxels
output: top voxel classifier array (the length of the array is the number of voxels)
******************************/
VoxelScore* ReadTopVoxelFile(const char* file, int n)
{
  int i;
  ifstream ifile(file);
  if (!ifile)
  {
    FATAL("file not found: "<<file);
  }
  VoxelScore* scores = new VoxelScore[n];
  for (i=0; i<n; i++)
  {
    ifile>>scores[i].vid>>scores[i].score;
  }
  ifile.close();
  return scores;
}

/******************************
Rearrange the raw data matrices to follow the top voxel order
intput: raw matrix array, the number of rows(#voxels), the number of columns(#TRs), the number of trials
output: rearrage matrix of the same array
*******************************/
void RearrangeMatrix(RawMatrix** r_matrices, VoxelScore* scores, int row, int col, int nSubs)
{
  int i, j;
  for (i=0; i<nSubs; i++)
  {
    float* curMat = new float[row*col];
    float* mat = r_matrices[i]->matrix;
    for (j=0; j<row; j++)
    {
      int rid = scores[j].vid;
      memcpy(curMat+j*col, mat+rid*col, sizeof(float)*col);
    }
    delete mat;
    r_matrices[i]->matrix = curMat;
  }
}

// row here is nTops, most of the time is function is not practical due to out of memory
float* GetInnerSimMatrix(int row, int col, int nTrials, Trial* trials, RawMatrix** r_matrices) // only compute the correlation among the selected voxels
{
  int i;
  float* values = new float[nTrials*row*row];
  float* simMatrix = new float[nTrials*nTrials];
  for (i=0; i<nTrials*nTrials; i++) simMatrix[i] = 0.0;
  for (i=0; i<nTrials; i++)
  {
    int sc = trials[i].sc;
    int ec = trials[i].ec;
    int sid = trials[i].sid;
    float* mat = r_matrices[sid]->matrix;
    float* buf = new float[row*col]; // col is more than what really need, just in case
    int ml = getBuf(sc, ec, row, col, mat, buf);  // get the normalized matrix, return the length of time points to be computed
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, row, row, ml, 1.0, buf, ml, buf, ml, 0.0, values+i*row*row, row);
    delete[] buf;
  }
  GetDotProductUsingMatMul(simMatrix, values, nTrials, row, row);
  delete[] values;
  return simMatrix;
}

// row here is nTops, get the inner product of vectors from start row(sr), last rowLength-length
float* GetPartialInnerSimMatrix(int row, int col, int nSubs, int nTrials, int sr, int rowLength, Trial* trials, RawMatrix** r_matrices) // only compute the correlation among the selected voxels
{
  int i;
  float* values = new float[nTrials*rowLength*row];
  float* simMatrix = new float[nTrials*nTrials];
  for (i=0; i<nTrials*nTrials; i++) simMatrix[i] = 0.0;
  for (i=0; i<nTrials; i++)
  {
    int sc = trials[i].sc;
    int ec = trials[i].ec;
    int sid = trials[i].sid;
    float* mat = r_matrices[sid]->matrix;
    //if (i==0 && sr==0) cout<<mat[1000*col]<<" "<<mat[1000*col+1]<<" "<<mat[1000*col+2]<<" "<<mat[1000*col+3]<<endl;
    //else if (i==0 && sr!=0) cout<<mat[0]<<" "<<mat[1]<<" "<<mat[2]<<" "<<mat[3]<<endl;
    float* buf = new float[row*col]; // col is more than what really need, just in case
    int ml = getBuf(sc, ec, row, col, mat, buf);  // get the normalized matrix, return the length of time points to be computed
    //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, step, row, ml, 1.0, buf+sr*ml, ml, buf, ml, 0.0, corrs, row);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rowLength, row, ml, 1.0, buf+sr*ml, ml, buf, ml, 0.0, values+i*rowLength*row, row);
    delete[] buf;
  }
  NormalizeCorrValues(values, nTrials, rowLength, row, nSubs);
  GetDotProductUsingMatMul(simMatrix, values, nTrials, rowLength, row);
  delete[] values;
  return simMatrix;
}

/******************************
compute the dot product of all-pair sub correlation vectors
*******************************/
void GetDotProductUsingMatMul(float* simMatrix, float* values, int nTrials, int nVoxels, int lengthPerCorrVector)
{
  int length = nVoxels*lengthPerCorrVector;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nTrials, nTrials, length, 1.0, values, length, values, length, 1.0, simMatrix, nTrials); // notice the latter 1.0 here, this is for accumulation
}

/***********************************
fisher transform and z-scored the correlation values across blocks (trials) within subject
this function is ad hoc, it needs all trials belonging to the same subjects be located together
size(values) = [nTrials, nVoxels*lengthPerCorrVector]
************************************/
void NormalizeCorrValues(float* values, int nTrials, int nVoxels, int lengthPerCorrVector, int nSubs)
{
  int length = nVoxels*lengthPerCorrVector;
  int trialsPerSub = nTrials / nSubs; // should be dividable
  int i, j;
  #pragma omp parallel for collapse(2) private(i,j)
  for (i=0; i<nSubs; i++) // do normalization subject by subject
  {
    for (j=0; j<length; j++)
    {
      int k;
      ALIGNED(64) float buf[trialsPerSub];
      for (k=0; k<trialsPerSub; k++)
      {
        buf[k] = fisherTransformation(values[i*trialsPerSub*length+k*length+j]);
        //cout<<values[i*trialsPerSub*length+k*length+j]<<" ";
        /*if (values[i*trialsPerSub*length+k*length+j]>=0)
          values[i*trialsPerSub*length+k*length+j] = log(values[i*trialsPerSub*length+k*length+j]+1);
        else
          values[i*trialsPerSub*length+k*length+j] = -log(-values[i*trialsPerSub*length+k*length+j]+1);
        buf[k] = values[i*trialsPerSub*length+k*length+j];*/
        //cout<<values[i*trialsPerSub*length+k*length+j]<<endl;
      }
      z_score(buf, trialsPerSub);
      for (k=0; k<trialsPerSub; k++)
      {
        values[i*trialsPerSub*length+k*length+j] = buf[k];
      }
    }
  }
}

/****************************
get the SVM training problem for precomputed model
input: the similairty matrix, the number of trials, the trial array, the number of training trials
output: the svm problem
*****************************/
SVMProblem* GetSVMTrainingSet(float* simMatrix, int nTrials, Trial* trials, int nTrainings)
{
  SVMProblem* prob = new SVMProblem();
  prob->l = nTrainings;
  prob->y = new double[nTrainings];
  prob->x = new SVMNode*[nTrainings];
  int i, j;
  for (i=0; i<nTrainings; i++)
  {
    prob->y[i] = trials[i].label;
    prob->x[i] = new SVMNode[nTrainings+2];
    prob->x[i][0].index = 0;
    prob->x[i][0].value = i+1;
    for (j=0; j<nTrainings; j++)
    {
      prob->x[i][j+1].index = j+1;
      prob->x[i][j+1].value = simMatrix[i*nTrials+j];
    }
    prob->x[i][j+1].index = -1;
  }
  return prob;
}
