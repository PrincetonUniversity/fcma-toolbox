#include "FileProcessing.h"
#include <nifti1_io.h>

/***************************************
Take a mask in (by default no masks), 
compute the correlation between two masks within brain, 
visualize the correlation results based on voxels in mask 1, 
generate a 4D nifti file, each time dimension depicts one voxel in mask 1, 
the 3D space dimensions depict the correlation between the corresonding voxel and the voxels in mask 2
input: the raw matrix data, the number of subjects, the first mask file, the second mask file, the dedicated block, the output file name
output: write the result to the nifti file
****************************************/
int VisualizeCorrelationWithMasks(RawMatrix* r_matrix, const char* maskFile1, const char* maskFile2, Trial trial, const char* output_file)
{
  int i, j;
  svm_set_print_string_function(&print_null);
  RawMatrix** masked_matrices1=NULL;
  RawMatrix** masked_matrices2=NULL;
  if (maskFile1!=NULL)
    masked_matrices1 = GetMaskedMatrices(r_matrices, nSubs, maskFile1);
  else
    masked_matrices1 = r_matrices;
  if (maskFile2!=NULL)
    masked_matrices2 = GetMaskedMatrices(r_matrices, nSubs, maskFile2);
  else
    masked_matrices2 = r_matrices;
  cout<<"masked matrices generating done!"<<endl;
  cout<<"#voxels for mask1: "<<masked_matrices1[0]->row<<" #voxels for mask2: "<<masked_matrices2[0]->row<<endl;
  float* simMatrix = new float[nTrials*nTrials];
  int corrRow = masked_matrices1[0]->row;
  //int corrCol = masked_matrices2[0]->row; // no use here
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
    delete tempSimMatrix;
    sr += rowLength;
  }
  SVMParameter* param = SetSVMParameter(4); // precomputed
  SVMProblem* prob = GetSVMTrainingSet(simMatrix, nTrials, trials, nTrials-nTests);
  struct svm_model *model = svm_train(prob, param);
  int nTrainings = nTrials-nTests;
  SVMNode* x = new SVMNode[nTrainings+2];
  double predict_distances[nTrials-nTrainings];
  bool predict_correctness[nTrials-nTrainings];
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
    int predict_label = predict_distances[i-nTrainings]>0?0:1;
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
  svm_free_and_destroy_model(&model);
  delete x;
  delete prob->y;
  for (i=0; i<nTrainings; i++)
  {
    delete prob->x[i];
  }
  delete prob->x;
  delete prob;
  svm_destroy_param(param);
  delete simMatrix;
  for (i=0; i<nSubs; i++)
  {
    delete masked_matrices1[i]->matrix;
    delete masked_matrices2[i]->matrix;
  }
  delete masked_matrices1;
  delete masked_matrices2;
  return result;
}
