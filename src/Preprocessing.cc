/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include <sstream>
#include <nifti1_io.h>
#include "common.h"
#include "ErrorHandling.h"
#include "Preprocessing.h"


/**************************
align multi-subjects, remove all-zero voxels, if all-zero in one subject, remove the corresponding voxel among all subjects
input: the raw matrix data structure array, the number of matrices (subjects)
output: the remaining number of voxels, remove voxels from raw matrix and location
***************************/
int AlignMatrices(RawMatrix** r_matrices, int nSubs, VoxelXYZ* pts)
{
  // - align multi-subjects
  // - remove all-zero voxels
  // - assume that all subjects have the same number of voxels
  int row = r_matrices[0]->row;
  bool flags[row];
  int i, j, k;
  for (i=0; i<row; i++)
  {
    flags[i] = true;
  }
  // get the zll-zero conditions, store in flags;
  // false means at least one subject contains all-zeros
  for (i=0; i<nSubs; i++)
  {
    int col = r_matrices[i]->col;
    float* mat = r_matrices[i]->matrix;
    for (j=0; j<row; j++)
    {
      bool flag = true;
      for (k=0; k<col; k++)
      {
        flag &= (mat[j*col+k]<=10.0);  // 10 is a threshold for "almost" all-zero
      }
      if (flag) flags[j] = false;
    }
  }
  int count=0;
  for (i=0; i<nSubs; i++) // remove the all-zero voxels
  {
    int col = r_matrices[i]->col;
    float* mat = r_matrices[i]->matrix;
    count = 0;
    for (j=0; j<row; j++)
    {
      if (flags[j])
      {
        memcpy(&(mat[count*col]), &(mat[j*col]), col*sizeof(float));
        count++;
      }
    }
    r_matrices[i]->row = count; // update the row information
  }
  count = 0;
  for (j=0; j<row; j++)
  {
    if (flags[j])
    {
      memcpy(&(pts[count]), &(pts[j]), 3*sizeof(int));
      count++;
    }
  }
  return count;
}

/**************************
align multi-subjects, using a file containing a binary vector that indicates whether or not to use a voxel
input: the raw matrix data structure array, the number of matrices (subjects), the binary vector file
output: the remaining number of voxels, remove voxels from raw matrix and location
***************************/
int AlignMatricesByFile(RawMatrix** r_matrices, int nSubs, const char* file, VoxelXYZ* pts)
{
#ifndef __MIC__
  // align multi-subjects, remove all-zero voxels, assume that all subjects have the same number of voxels  
  int row = r_matrices[0]->row; // originally is the total number of voxels x*y*z
  int i, j;
  int count=0;
  nifti_image* nim;
  nim = nifti_image_read(file, 1);  // for inputting mask file
  short* data = NULL;
  switch (nim->datatype)  // now only get one type
  {
    case DT_SIGNED_SHORT:
      data = (short*)nim->data;
      break;
    default:
      FATAL("wrong data type of mask file!");
  }
  assert(data);
  for (i=0; i<nSubs; i++) // remove the all-zero voxels
  {
    int col = r_matrices[i]->col;
    float* mat = r_matrices[i]->matrix;
    count = 0;
    for (j=0; j<row; j++)
    {
      if (data[j])
      {
        memcpy(&(mat[count*col]), &(mat[j*col]), col*sizeof(float));
        count++;
      }
    }
    r_matrices[i]->row = count; // update the row information
  }
  count = 0;
  for (j=0; j<row; j++)
  {
    if (data[j])
    {
      memcpy(&(pts[count]), &(pts[j]), 3*sizeof(int));
      count++;
    }
  }
  nifti_image_free(nim);
  return count;
#else
  return 0;
#endif
}

/*********************************
put some trials to the end to be left out when doing voxel selection
input: the trial array, the number of trials, the starting leave-out trial id, the number of trials that left out
output: the leave-out subject's data is put to the tail of the trial array
**********************************/
void leaveSomeTrialsOut(Trial* trials, int nTrials, int tid, int nLeaveOut)
{
  Trial temp[nLeaveOut];
  int i, newIndex=0, count=0;
  for (i=0; i<nTrials; i++)
  {
    if (i>=tid && i<tid+nLeaveOut)
    {
      temp[count] = trials[i];
      count++;
    }
    else  // the trial is not taken out
    {
      trials[newIndex] = trials[i];
      newIndex++;
    }
  }
  count=0;
  for (i=nTrials-nLeaveOut; i<nTrials; i++)
  {
    trials[i] = temp[count];
    count++;
  }
}

/****************************************
Fisher transform the correlation values (coefficients) then z-scored them across blocks
input: the correlation matrix array, the length of this array (the number of blocks), the number of subjects
output: update values to the matrices
*****************************************/
void corrMatPreprocessing(CorrMatrix** c_matrices, int n, int nSubs)
{
  int row = c_matrices[0]->step;
  int col = c_matrices[0]->nVoxels; // assume that all correlation matrices have the same size
  int i;
  if (n%nSubs!=0)
  {
    FATAL("number of blocks in every subject must be the same");
  }
  int nPerSub = n/nSubs;
  // assume that the blocks belonged to the same subject are placed together
  // for each subject, go through the available voxel pairs;
  // then for each pairs, Fisher transform it and z-score within subject
  // doing this can make better cache usage than going through voxel pairs in the outtest loop
  for (int k=0; k<nSubs; k++)
  {
    #pragma omp parallel for private(i)
    for (i=0; i<row*col; i++)
    {
      int j;
      ALIGNED(64) float buf[nPerSub];
      #pragma simd
      for (j=0; j<nPerSub; j++)
      {
        buf[j] = fisherTransformation(c_matrices[k*nPerSub+j]->matrix[i]);
      }
      if (nPerSub>1)  // nPerSub==1 results in 0
        z_score(buf, nPerSub);
      for (j=0; j<nPerSub; j++)
      {
        c_matrices[k*nPerSub+j]->matrix[i] = buf[j];
      }
    }
  }
}

/****************************************
average the activation values by blocks then z-scored them across blocks within subjects
input: the raw matrix array, the length of this array (the number of subjects), the number of blocks per subject, the trials
output: the average values of all subjects after z-scoring
*****************************************/
RawMatrix** rawMatPreprocessing(RawMatrix** r_matrices, int n, int nTrialsPerSub, Trial* trials)
{
  int row = r_matrices[0]->row; // assume that all correlation matrices have the same size
  RawMatrix** avg_matrices = new RawMatrix*[n];
  int i, j, k;
  for (i=0; i<n; i++)
  {
    avg_matrices[i] = new RawMatrix();
    avg_matrices[i]->sid = i;
    avg_matrices[i]->row = row; // the number of voxels
    avg_matrices[i]->col = nTrialsPerSub; // the number of blocks per subject
    avg_matrices[i]->nx = r_matrices[i]->nx;
    avg_matrices[i]->ny = r_matrices[i]->ny;
    avg_matrices[i]->nz = r_matrices[i]->nz;
    avg_matrices[i]->matrix = new float[row*nTrialsPerSub];
    for (j=0; j<row; j++)
    {
      for (k=0; k<nTrialsPerSub; k++)
      {
        avg_matrices[i]->matrix[j*nTrialsPerSub+k] = getAverage(r_matrices[i], trials[k], j);
      }
      z_score(&(avg_matrices[i]->matrix[j*nTrialsPerSub]), nTrialsPerSub);
    }
  }
  return avg_matrices;
}

// average the activition values of a block
float getAverage(RawMatrix* r_matrix, Trial trial, int vid)
{
  if (vid == -1)
  {
    return 0.0;
  }
  int i;
  float result = 0.0;
  int col = r_matrix->col;
  for (i=trial.sc; i<=trial.ec; i++)
  {
    result += float(r_matrix->matrix[vid*col+i]);
  }
  result /= (trial.ec-trial.sc+1);
  return result;
}

/****************************************
permute voxels within subject based on an input file or a given seed randomization
input: the raw matrix array, the length of this array (the number of subjects), the randomization seed, the permuting file
output: voxels have been randomly permuted
*****************************************/
void MatrixPermutation(RawMatrix** r_matrices, int nSubs, unsigned int seed, const char* permute_book_file)
{

  int row = r_matrices[0]->row;
  int col = r_matrices[0]->col;
    
  assert(row>0 && col>0);
  
  int i, j;
  float buf[row];
  int index[row];
  std::ifstream ifile;
    
  if (permute_book_file)  // use permute book
  {
    ifile.open(permute_book_file, std::ios::in);
    if (!ifile)
    {
      FATAL("file not found: "<<permute_book_file);
    }
  }
  else // use build-in random
  {
    srand(seed);
    for (i=0; i<row; i++) index[i]=i;
  }
  int k;
  for (i=0; i<nSubs; i++)
  {
    if (permute_book_file)
    {
      for (j=0; j<row; j++) ifile>>index[j];
    }
    else if (row > 1)
    {
        /* shuffle function most effective if row << RAND_MAX */
         for (size_t r = 0; r < row - 1; r++)
        {
            size_t j = r + rand() / (RAND_MAX / (row - r) + 1);
            int t = index[j];
            index[j] = index[r];
            index[r] = t;
        }
    }
    for (j=0; j<row; j++)
    {
      k = index[j];
      memcpy((void*)buf, (const void*)&(r_matrices[i]->matrix[j*col]), col*sizeof(float));
      memcpy((void*)&(r_matrices[i]->matrix[j*col]), (const void*)&(r_matrices[i]->matrix[k*col]), col*sizeof(float));
      memcpy((void*)&(r_matrices[i]->matrix[k*col]), (const void*)buf, col*sizeof(float));
    }
  }
  if (permute_book_file)
  {
    ifile.close();
  }
}

/****************************************
use the trials information to filter the given raw matrices and normalize it for correlation computation
input: the raw matrix array, the buffer trial array storing old trial info, the trial array, the length of raw matrix array (the number of subjects), the length of trials array (the number of trials)
output: the matrix array is rewritten to be one by one submatrix instead of one big matrix and the trial array is modified accordingly
*****************************************/
TrialData* PreprocessMatrices(RawMatrix** matrices, Trial* trials, int nSubs, int nTrials)
{
  assert(nTrials>1);
  assert(matrices);
  assert(matrices[0]);
    
  TrialData* td = new TrialData(nTrials, matrices[0]->row);
  // making this an array allows for variable-sized trials (ie blocks)
  td->trialLengths = new int[nTrials];
  td->scs = new int[nTrials];
  int total_cols=0;
  // get total TRs of each subject
  // note that nTrials is across subjects
  // that is, nTrials is nSubs * trials (ie blocks) per subject
  for (int i=0; i<nTrials; i++)
  {
    td->scs[i] = total_cols;
    total_cols += trials[i].ec-trials[i].sc+1;
    td->trialLengths[i] = trials[i].ec-trials[i].sc+1;
  }
  // nCols was nTrials * trs_per_block when trialLengths were constant
  // now use the cumulative sum of trialLengths from above
  td->nCols = total_cols;
  
  size_t dataSize = sizeof(float) * (size_t)td->nCols * (size_t)td->nVoxels;

  std::cout << "allocating raw data buffer bytes: " << dataSize << std::endl << std::flush;
 
  td->data = (float*)_mm_malloc(dataSize, 64);
  assert(td->data);
  int cur_cols[nTrials];
  cur_cols[0] = 0;
  for (int i=1; i<nTrials; i++)
  {
    int sc= trials[i-1].sc;
    int ec= trials[i-1].ec;
    int delta_col = ec-sc+1;
    cur_cols[i] = cur_cols[i-1]+delta_col;
  }
  // for all trials (ie blocks * subjects)
  #pragma omp parallel for
  for (int i=0; i<nTrials; i++)
  {
    int sid = trials[i].sid;
    int row = matrices[sid]->row;
    int col = matrices[sid]->col;
    float* matrix = matrices[sid]->matrix;
    float* buf = td->data + (size_t)cur_cols[i] * (size_t)td->nVoxels;
    int sc = trials[i].sc;
    int ec = trials[i].ec;
    int delta_col = ec-sc+1;
    assert(delta_col > 0);
    
    // for all voxels (rows)
    for (int r=0; r<row; r++)
    {
      double mean=0.0f, sd=0.0f;  // float here is not precise enough to handle
      // normalization 1: get mean+sd for a particular block within a particular subject
      for (int c=sc; c<=ec; c++)
      {
        mean += (double)matrix[r*col+c];
        sd += (double)matrix[r*col+c] * matrix[r*col+c]; // convert to double to avoid overflow
      }
      mean /= delta_col;
      sd = sd - delta_col * mean * mean;
      sd = sqrt(sd);
      ALIGNED(64) float inv_sd_f;
      ALIGNED(64) float mean_f=mean;  // for vectorization
      if (sd == 0.0f)  // leave the numbers as they are
      {
        inv_sd_f=0.0f;
      }
      else
      {
        inv_sd_f=1.0f/sd;  // do time-comsuming division once
      }
      
      // normalization 2: subtract mean and divide by sd calculated above
      #pragma simd
      for (int c=sc; c<=ec; c++)
      {
        // if sd is zero, a "nan" appears
        buf[r*delta_col+c-sc] = (matrix[r*col+c] - mean_f) * inv_sd_f;
      }
      // so buffer data is organized as trialLength vectors (trs in a block)
    }
  }
  return td;
}
