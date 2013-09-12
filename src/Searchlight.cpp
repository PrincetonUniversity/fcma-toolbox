/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "Searchlight.h"
#include "SVMClassification.h"
#include "Classification.h"   // for distance ratio
#include "Scheduler.h"
#include "Preprocessing.h"    // for z-score across blocks within subjects
#include "FileProcessing.h"   // for generating masked matrices, writing nifti files
#include "common.h"

// don't forget shift 2!! we can speficy the shifting in the block file, but we don't use shifting for face/scene dataset

/***********************************
main function of this file, do the traditional feature selection and classification
input: the averaged raw matrix array, the number of subjects, the trials, the number of trials, the number of test samples, the number of folds in the svm-cv, the voxel location information, the top voxel file name, the mask file name
output: write the voxels in decreasing order of classification accuracy to the top voxel file\
************************************/
void Searchlight(RawMatrix** avg_matrices, int nSubs, Trial* trials, int nTrials, int nTests, int nFolds, Point* pts, const char* topVoxelFile, const char* maskFile)
{
  RawMatrix** masked_matrices=NULL;
  Point* masked_pts=NULL;
  if (maskFile!=NULL)
  {
    masked_matrices = GetMaskedMatrices(avg_matrices, nSubs, maskFile);
    masked_pts = GetMaskedPts(pts, masked_matrices[0]->row, maskFile);
  }
  else
  {
    masked_matrices = avg_matrices;
    masked_pts = pts;
  }
  int i;
  int row = masked_matrices[0]->row;  // assume all elements in r_matrices array have the same row
  cout<<"#voxels for mask: "<<row<<endl;
  VoxelScore* scores = GetSearchlightSVMPerformance(masked_matrices, trials, nTrials, nTests, nFolds, masked_pts);
  sort(scores, scores+row, cmp);
  char fullfilename[MAXFILENAMELENGTH];
  sprintf(fullfilename, "%s", topVoxelFile);
  strcat(fullfilename, "_list.txt");
  ofstream ofile(fullfilename);
  for (i=0; i<row; i++)
  {
    ofile<<scores[i].vid<<" "<<scores[i].score<<endl;
  }
  ofile.close();
  int* data_ids = (int*)GenerateNiiDataFromMask(maskFile, scores, row, DT_SIGNED_INT);
  sprintf(fullfilename, "%s", topVoxelFile);
  strcat(fullfilename, "_seq.nii.gz");
  WriteNiiGzData(fullfilename, maskFile, (void*)data_ids, DT_SIGNED_INT);
  float* data_scores = (float*)GenerateNiiDataFromMask(maskFile, scores, row, DT_FLOAT32);
  sprintf(fullfilename, "%s", topVoxelFile);
  strcat(fullfilename, "_score.nii.gz");
  WriteNiiGzData(fullfilename, maskFile, (void*)data_scores, DT_FLOAT32);
  delete[] scores;
  if (maskFile!=NULL)
    delete masked_pts;
}

/****************************************
get the SVM results of classifying activation vectors of two categories for every voxel
the linear kernel of libSVM is applied here
input: the average activation matrix array, the blocks(trials), the number of blocks, the number of test samples, the number of folds in the cross validation, the location info
output: a list of voxels' scores in terms of SVM accuracies
*****************************************/
VoxelScore* GetSearchlightSVMPerformance(RawMatrix** avg_matrices, Trial* trials, int nTrials, int nTests, int nFolds, Point* pts)  //classifiers for a r_matrix array
{
  svm_set_print_string_function(&print_null);
  int row = avg_matrices[0]->row;  // assume all elements in r_matrices array have the same row
  VoxelScore* score = new VoxelScore[row];  // get step voxels classification accuracy here
  int i, j;
  for (i=0; i<row; i++)
  {
    SVMProblem* prob = GetSearchlightSVMProblem(avg_matrices, trials, i, nTrials-nTests, pts);
    if (i%1000==0) cout<<i<<" ";
    //if (me==0) PrintSVMProblem(prob);
    SVMParameter* param = SetSVMParameter(0); // 2 for RBF kernel
    (score+i)->vid = i;
    (score+i)->score = DoSVM(nFolds, prob, param);
    delete param;
    delete prob->y;
    for (j=0; j<nTrials-nTests; j++)
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
  return score;
}

/*****************************************
generate a SVM classification problem
input: the average activation matrix array, the number of blocks, the blocks, the current voxel id, the number of training samples, the location info
output: the SVM problem described in the libSVM recognizable format
******************************************/
SVMProblem* GetSearchlightSVMProblem(RawMatrix** avg_matrices, Trial* trials, int curVoxel, int nTrainings, Point* pts)
{
  SVMProblem* prob = new SVMProblem();
  prob->l = nTrainings;
  prob->y = new double[nTrainings];
  prob->x = new SVMNode*[nTrainings];
  int nVoxels = avg_matrices[0]->row;
  int* voxels = GetSphere(curVoxel, nVoxels, pts);
  int nSphereVoxels = 33;
  #pragma omp parallel for
  for (int i=0; i<nTrainings; i++)
  {
    //cout<<trials[i].tid<<" "<<trials[i].sid<<" "<<trials[i].label<<" "<<trials[i].sc<<" "<<trials[i].ec<<endl;
    prob->y[i] = trials[i].label;
    prob->x[i] = new SVMNode[nSphereVoxels+1];
    int sid = trials[i].sid;
    int j;
    for (j=0; j<nSphereVoxels; j++)
    {
      prob->x[i][j].index = j+1;
      int col = avg_matrices[sid]->col;
      int offset = trials[i].tid_withinsubj;
      if (voxels[j]!=-1)
      {
        prob->x[i][j].value = avg_matrices[sid]->matrix[voxels[j]*col+offset];
      }
      else
      {
        prob->x[i][j].value = 0;
      }
      //cout<<voxels[j]; getchar();
    }
    prob->x[i][j].index = -1;
  }
  delete[] voxels;
  return prob;
}

// get the nearby voxel ids of a given voxel, within the radius 2 here
int* GetSphere(int voxelId, int nVoxels, Point* pts)
{
  int x = pts[voxelId].x;
  int y = pts[voxelId].y;
  int z = pts[voxelId].z;
  int* results = new int[33];
  results[0] = GetPoint(x-1, y-1, z-1, nVoxels, pts);
  results[1] = GetPoint(x, y-1, z-1, nVoxels, pts);
  results[2] = GetPoint(x+1, y-1, z-1, nVoxels, pts);
  results[3] = GetPoint(x-1, y, z-1, nVoxels, pts);
  results[4] = GetPoint(x, y, z-1, nVoxels, pts);
  results[5] = GetPoint(x+1, y, z-1, nVoxels, pts);
  results[6] = GetPoint(x-1, y+1, z-1, nVoxels, pts);
  results[7] = GetPoint(x, y+1, z-1, nVoxels, pts);
  results[8] = GetPoint(x+1, y+1, z-1, nVoxels, pts);
  results[9] = GetPoint(x-1, y-1, z, nVoxels, pts);
  results[10] = GetPoint(x, y-1, z, nVoxels, pts);
  results[11] = GetPoint(x+1, y-1, z, nVoxels, pts);
  results[12] = GetPoint(x-1, y, z, nVoxels, pts);
  results[13] = GetPoint(x, y, z, nVoxels, pts);
  results[14] = GetPoint(x+1, y, z, nVoxels, pts);
  results[15] = GetPoint(x-1, y+1, z, nVoxels, pts);
  results[16] = GetPoint(x, y+1, z, nVoxels, pts);
  results[17] = GetPoint(x+1, y+1, z, nVoxels, pts);
  results[18] = GetPoint(x-1, y-1, z+1, nVoxels, pts);
  results[19] = GetPoint(x, y-1, z+1, nVoxels, pts);
  results[20] = GetPoint(x+1, y-1, z+1, nVoxels, pts);
  results[21] = GetPoint(x-1, y, z+1, nVoxels, pts);
  results[22] = GetPoint(x, y, z+1, nVoxels, pts);
  results[23] = GetPoint(x+1, y, z+1, nVoxels, pts);
  results[24] = GetPoint(x-1, y+1, z+1, nVoxels, pts);
  results[25] = GetPoint(x, y+1, z+1, nVoxels, pts);
  results[26] = GetPoint(x+1, y+1, z+1, nVoxels, pts);
  results[27] = GetPoint(x-2, y, z, nVoxels, pts);
  results[28] = GetPoint(x+2, y, z, nVoxels, pts);
  results[29] = GetPoint(x, y-2, z, nVoxels, pts);
  results[30] = GetPoint(x, y+2, z, nVoxels, pts);
  results[31] = GetPoint(x, y, z-2, nVoxels, pts);
  results[32] = GetPoint(x, y, z+2, nVoxels, pts);
  return results;
}


//get the voxel id given the 3d coordinates
int GetPoint(int x, int y, int z, int nVoxels, Point* pts)
{
  int i;
  for (i=0; i<nVoxels; i++)
  {
    if (pts[i].x==x && pts[i].y==y && pts[i].z==z)
    {
      return i;
    }
    if (pts[i].x>x && pts[i].y>y && pts[i].z>z)
    {
      return -1;
    }
  }
  return -1;
}
