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
#include "CustomizedMatrixMultiply.h"
#include <nifti1_io.h>

using namespace std;


/***************************************
Get two parts of the brain to compute the correlation and then use the
correlation vectors to predict
input: the masked activation matrix arrays, the number of subjects, the number of
subjects, the number of
blocks(trials), the blocks, the number of test samples
output: the results are displayed on the screen and returned
****************************************/
int SVMPredictCorrelationWithMasks(RawMatrix** masked_matrices1,
                                   RawMatrix** masked_matrices2, int nSubs,
                                   int nTrials, Trial* trials, int nTests,
                                   int is_quiet_mode) {
#ifndef __MIC__
  int i, j, k;
  svm_set_print_string_function(&print_null);
#if __MEASURE_TIME__
  float t_sim = 0.0f, t_train = 0.0f;
  struct timeval start, end;
  gettimeofday(&start, 0);
#endif
  float* simMatrix = new float[(CMM_INT)nTrials * nTrials];
  int corrRow = masked_matrices1[0]->row;
  memset((void*)simMatrix, 0, nTrials * nTrials * sizeof(float));
  CMM_INT sr = 0, rowLength = 100;
  int result = 0;
  while (sr < corrRow) {
    if (rowLength >= corrRow - sr) {
      rowLength = corrRow - sr;
    }
    float* tempSimMatrix =
        GetPartialInnerSimMatrixWithMasks(nSubs, nTrials, sr, rowLength, trials,
                                          masked_matrices1, masked_matrices2);
    for (i = 0; i < (CMM_INT)nTrials * nTrials; i++) simMatrix[i] += tempSimMatrix[i];
    delete[] tempSimMatrix;
    sr += rowLength;
  }
#if __MEASURE_TIME__
  gettimeofday(&end, 0);
  t_sim = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 0.000001;
  cout << "similarity (kernel) matrix computation done! Takes " << t_sim << "s"
       << endl;
  gettimeofday(&start, 0);
#endif
// LibSVM
  SVMParameter* param = SetSVMParameter(PRECOMPUTED);  // LINEAR or PRECOMPUTED
  SVMProblem* prob =
      GetSVMTrainingSet(simMatrix, nTrials, trials, nTrials - nTests);
  struct svm_model* model = svm_train(prob, param);
#if __MEASURE_TIME__
  gettimeofday(&end, 0);
  t_train =
      end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 0.000001;
  cout << "LibSVM training done! Takes " << t_train << "s" << endl;
#endif
  int nTrainings = nTrials - nTests;
  SVMNode* x = new SVMNode[nTrainings + 2];
  double predict_distances[nTrials - nTrainings];
  memset(predict_distances, 0,
         (nTrials - nTrainings) * sizeof(double));  // bds init
  bool predict_correctness[nTrials - nTrainings];
  memset(predict_correctness, false,
         (nTrials - nTrainings) * sizeof(bool));  // bds init
  for (i = nTrainings; i < nTrials; i++) {
    x[0].index = 0;
    x[0].value = i - nTrainings + 1;
    for (j = 0; j < nTrainings; j++) {
      x[j + 1].index = j + 1;
      x[j + 1].value = simMatrix[i * nTrials + j];
    }
    x[j + 1].index = -1;
    predict_distances[i - nTrainings] = svm_predict_distance(model, x);
    // int predict_label = predict_distances[j-nTrainings]>0?0:1;
    int predict_label = int(svm_predict(model, x));
    if (trials[i].label == predict_label) {
      result++;
      predict_correctness[i - nTrainings] = true;
    } else {
      predict_correctness[i - nTrainings] = false;
    }
  }
  if (!is_quiet_mode) {
    cout << "blocking testing confidence:" << endl;
    for (i = nTrainings; i < nTrials; i++) {
      cout << fabs(predict_distances[i - nTrainings]) << " (";
      if (predict_correctness[i - nTrainings]) {
        cout << "Correct) ";
      } else {
        cout << "Incorrect) ";
      }
    }
    cout << endl;
  }
  svm_free_and_destroy_model(&model);
  delete[] x;
  delete prob->y;
  for (i = 0; i < nTrainings; i++) {
    delete prob->x[i];
  }
  delete prob->x;
  delete prob;
  svm_destroy_param(param);
  // LibSVM done
  /*int nTrainingSamples = nTrials - nTests;
  float* trainingData = new float[nTrainingSamples*nTrainingSamples];
  for (j=0 ; j<nTrainingSamples; j++) {
    for (k=0; k<nTrainingSamples; k++) {
      trainingData[j*nTrainingSamples+k] = simMatrix[j*nTrials+k]*.001f;
    }
  }
  float* labels = new float[nTrainingSamples];
  for (j=0 ; j<nTrainingSamples; j++) {
    labels[j] = trials[j].label == 0 ? -1.0f : 1.0f;
  }
  Kernel_params kp;
  kp.gamma = 0;
  kp.coef0 = 0;
  kp.degree = 3;
  // kp.b doesn't need to be preset
  kp.kernel_type = "precomputed";
  float cost = 10.0f;
  SelectionHeuristic heuristicMethod = ADAPTIVE;
  float tolerance = 1e-3f;
  float epsilon = 1e-5f;
  PhiSVMModel* phiSVMModel = performTraining(trainingData, nTrainingSamples,
                  nTrainingSamples, labels,
                  &kp, cost, heuristicMethod, epsilon, tolerance, NULL, NULL);
  #if __MEASURE_TIME__
    gettimeofday(&end, 0);
    t_train =
        end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 0.000001;
    cout << "phiSVM training done! Takes " << t_train << "s" << endl;
  #endif
  float* testData = new float[nTests*nTrainingSamples];
  for (j=0 ; j<nTests; j++) {
    for (k=0; k<nTrainingSamples; k++) {
      testData[j*nTrainingSamples+k] = simMatrix[(j+nTrainingSamples)*nTrials+k]*.001f;
    }
  }
  float* testLabels = new float[nTests];
  for (j=0 ; j<nTests; j++) {
    testLabels[j] = trials[j+nTrainingSamples].label == 0 ? -1.0f : 1.0f;
  }
  float* result_vec;
  performClassification(testData, nTests,
                        nTrainingSamples, &kp, &result_vec, phiSVMModel);
  //delete phiSVMModel;
  for (j = 0; j < nTests; j++) {
    result =
        (testLabels[j] == 1 && result_vec[j] >= 0) || (testLabels[j] == -1 && result_vec[j] < 0)
            ? result + 1
            : result;
  }
  if (!is_quiet_mode) {
    using std::cout;
    using std::endl;

    cout << "blocking testing confidence:" << endl;
    for (j = 0; j < nTests; j++) {
      cout << fabs(result_vec[j]) << " (";
      if ((testLabels[j] == 1 && result_vec[j] >= 0) || (testLabels[j] == -1 && result_vec[j] < 0)) {
        cout << "Correct) ";
      } else {
        cout << "Incorrect) ";
      }
    }
    cout << endl;
  }*/
  // phiSVM done
  delete[] simMatrix;
  return result;
#else
  return 0;
#endif
}

/***********************************************
Get the inner product of vectors from start row(sr), last rowLength-length
input: the number of subjects, the number of blocks, the start row, the number
of voxels of masked matrix one that involved in the computing, the trials
information, the first masked data array, the second masked data array
output: the partial similarity matrix based on the selected rows of first
matrices and the whole second matrices
************************************************/
float* GetPartialInnerSimMatrixWithMasks(
    int nSubs, int nTrials, int sr, int rowLength, Trial* trials,
    RawMatrix** masked_matrices1,
    RawMatrix** masked_matrices2)  // compute the correlation between masked
                                   // matrices
{
  int i;
  int row1 = masked_matrices1[0]->row;
  int row2 = masked_matrices2[0]->row;  // rows should be the same across
                                        // subjects since we are using the same
                                        // mask file to filter out voxels
  float* values = new float[(CMM_INT)nTrials * rowLength * row2];
  float* simMatrix = new float[nTrials * nTrials];
  memset((void*)simMatrix, 0, nTrials * nTrials * sizeof(float));
  for (i = 0; i < nTrials; i++) {
    int sc = trials[i].sc;
    int ec = trials[i].ec;
    int sid = trials[i].sid;
    int col = masked_matrices1[sid]->col;  // the column of 1 and 2 should be
                                           // the same, i.e. the number of TRs
                                           // of a block; columns may be
                                           // different, since different
                                           // subjects have different TRs
    float* mat1 = masked_matrices1[sid]->matrix;
    float* mat2 = masked_matrices2[sid]->matrix;
    float* buf1 = new float[(CMM_INT)row1 * col];  // col is more than what really need,
                                          // just in case
    float* buf2 = new float[(CMM_INT)row2 * col];  // col is more than what really need,
                                          // just in case
    int ml1 = getBuf(sc, ec, row1, col, mat1, buf1);  // get the normalized
                                                      // matrix, return the
                                                      // length of time points
                                                      // to be computed
    int ml2 = getBuf(sc, ec, row2, col, mat2, buf2);  // get the normalized
                                                      // matrix, return the
                                                      // length of time points
                                                      // to be computed, m1==m2
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rowLength, row2, ml1,
                1.0, buf1 + sr * ml1, ml1, buf2, ml2, 0.0,
                values + (CMM_INT)i * rowLength * row2, row2);
    delete[] buf1;
    delete[] buf2;
  }
  NormalizeCorrValues(values, nTrials, rowLength, row2, nSubs);
  GetDotProductUsingMatMul(simMatrix, values, nTrials, rowLength, row2);
  delete[] values;
  return simMatrix;
}

/***************************************
Get one part of the brain to compute the averaged activation and then use the
normalized activation vectors to predict
input: the raw activation matrix array, the number of voxels, the number of
subjects, the ROI mask file, the number of blocks(trials), the blocks, the
number of test samples
output: the results are displayed on the screen and returned
****************************************/
int SVMPredictActivationWithMasks(RawMatrix** avg_matrices, int nSubs,
                                  const char* maskFile, int nTrials,
                                  Trial* trials, int nTests,
                                  int is_quiet_mode) {
#ifndef __MIC__
  int i, j;
  int nTrainings = nTrials - nTests;
  SVMParameter* param = SetSVMParameter(LINEAR);  // LINEAR or PRECOMPUTED
  SVMProblem* prob = new SVMProblem();
  prob->l = nTrainings;
  prob->y = new schar[nTrainings];
  prob->x = new SVMNode* [nTrainings];
  svm_set_print_string_function(&print_null);

  RawMatrix** masked_matrices = NULL;
  if (maskFile != NULL)
    masked_matrices = GetMaskedMatrices(avg_matrices, nSubs, maskFile, true);
  else
    masked_matrices = avg_matrices;
  cout << "masked matrices generating done!" << endl;
  cout << "#voxels for mask " << masked_matrices[0]->row << endl;
  int nVoxels = masked_matrices[0]->row;
  for (i = 0; i < nTrainings; i++) {
    int sid = trials[i].sid;
    prob->y[i] = trials[i].label;
    prob->x[i] = new SVMNode[nVoxels + 1];
    for (j = 0; j < nVoxels; j++) {
      prob->x[i][j].index = j + 1;
      int col = masked_matrices[sid]->col;
      int offset = trials[i].tid_withinsubj;
      prob->x[i][j].value = masked_matrices[sid]->matrix[j * col + offset];
    }
    prob->x[i][j].index = -1;
  }
  struct svm_model* model = svm_train(prob, param);
  SVMNode* x = new SVMNode[nVoxels + 1];
  double predict_distances[nTrials - nTrainings];
  memset(predict_distances, 0,
         (nTrials - nTrainings) * sizeof(double));  // bds init
  bool predict_correctness[nTrials - nTrainings];
  memset(predict_correctness, false,
         (nTrials - nTrainings) * sizeof(bool));  // bds init
  int result = 0;
  for (i = nTrainings; i < nTrials; i++) {
    int sid = trials[i].sid;
    for (j = 0; j < nVoxels; j++) {
      x[j].index = j + 1;
      int col = masked_matrices[sid]->col;
      int offset = trials[i].tid_withinsubj;
      x[j].value = masked_matrices[sid]->matrix[j * col + offset];
    }
    x[j].index = -1;
    predict_distances[i - nTrainings] = svm_predict_distance(model, x);
    // int predict_label = predict_distances[j-nTrainings]>0?0:1;
    int predict_label = int(svm_predict(model, x));
    if (trials[i].label == predict_label) {
      result++;
      predict_correctness[i - nTrainings] = true;
    } else {
      predict_correctness[i - nTrainings] = false;
    }
  }
  if (!is_quiet_mode) {
    cout << "blocking testing confidence:" << endl;
    for (i = nTrainings; i < nTrials; i++) {
      cout << fabs(predict_distances[i - nTrainings]) << " (";
      if (predict_correctness[i - nTrainings]) {
        cout << "Correct) ";
      } else {
        cout << "Incorrect) ";
      }
    }
    cout << endl;
  }
  svm_free_and_destroy_model(&model);
  delete[] x;
  delete[] prob->y;
  for (i = 0; i < nTrainings; i++) {
    delete prob->x[i];
  }
  delete[] prob->x;
  delete prob;
  svm_destroy_param(param);
  for (i = 0; i < nSubs; i++) {
    delete masked_matrices[i]->matrix;
  }
  delete masked_matrices;
  return result;
#else
  return 0;
#endif
}
