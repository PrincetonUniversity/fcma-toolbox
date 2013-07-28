#include "Preprocessing.h"
#include "common.h"
#include <sstream>
#include <nifti1_io.h>

/**************************
align multi-subjects, remove all-zero voxels, if all-zero in one subject, remove the corresponding voxel among all subjects
input: the raw matrix data structure array, the number of matrices (subjects)
output: the remaining number of voxels, remove voxels from raw matrix and location
***************************/
int AlignMatrices(RawMatrix** r_matrices, int nSubs, Point* pts)
{
  // align multi-subjects, remove all-zero voxels, assume that all subjects have the same number of voxels  
  int row = r_matrices[0]->row;
  bool flags[row];  // can use a variable to define an array
  int i, j, k;
  for (i=0; i<row; i++)
  {
    flags[i] = true;
  }
  for (i=0; i<nSubs; i++) // get the zll-zero conditions, store in flags, false means at least one subject contains all-zeros
  {
    int col = r_matrices[i]->col;
    float* mat = r_matrices[i]->matrix;
    for (j=0; j<row; j++)
    {
      bool flag = true;
      for (k=0; k<col; k++)
      {
        flag &= (mat[j*col+k]<=10.0);  // 2 is a threshold for "almost" all-zero
      }
      if (flag) flags[j] = false;
    }
  }
  int count=0;
  /*ofstream ofile("masks.txt");  // for outputting mask file
  for (i=0; i<row; i++)
  {
    if (flags[i]) ofile<<"1 ";
    else ofile<<"0 ";
  }
  ofile.close();*/
  //exit(1);
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
  // one-time rearrange the location information and new location file writing
  /*count = 0;
  for (j=0; j<row; j++)
  {
    if (flags[j])
    {
      memcpy(&(pts[count]), &(pts[j]), 3*sizeof(int));
      count++;
    }
  }
  FILE *fp = fopen("/memex/yidawang/neuroscience/data/Abby/Parity_Magnitude/location_new.bin", "wb");
  fwrite((void*)&count, sizeof(int), 1, fp);
  fwrite((void*)pts, sizeof(int), 3*count, fp);
  fclose(fp);
  exit(1);*/
  return count;
}

/**************************
align multi-subjects, using a file containing a binary vector that indicates whether or not to use a voxel
input: the raw matrix data structure array, the number of matrices (subjects), the binary vector file
output: the remaining number of voxels, remove voxels from raw matrix and location
***************************/
int AlignMatricesByFile(RawMatrix** r_matrices, int nSubs, const char* file, Point* pts)
{
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
      cerr<<"wrong data type of mask file!"<<endl;
      exit(1);
  }
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
RT regression, Fisher transform the correlation values (coefficients) then z-scored them across blocks
input: the correlation matrix array, the length of this array (the number of blocks), the number of subjects
output: update values to the matrices
*****************************************/
void corrMatPreprocessing(CorrMatrix** c_matrices, int n, int nSubs)
{
  int row = c_matrices[0]->step;
  int col = c_matrices[0]->nVoxels; // assume that all correlation matrices have the same size
  int i;
  #pragma omp parallel for private(i)
  for (i=0; i<row*col; i++)
  {
    float buf[n];
    int j, k;
    // need to do RT regression and get z-scored subject by subject
    for (k=0; k<nSubs; k++)
    {
      int count = 0;
      for (j=0; j<n; j++)
      {
        if (c_matrices[j]->sid == k)
        {
          buf[count] = c_matrices[j]->matrix[i];
          count++;
        }
      }
      /*for (j=0; j<n; j++)
      {
        if (c_matrices[j]->sid == k)
        {
          buf[count] = fisherTransformation(c_matrices[j]->matrix[i]);
          count++;
        }
      }*/
      for (j=0; j<count; j++)
      {
        buf[j] = fisherTransformation(buf[j]);
        //buf[j] = fabs(buf[j]);
      }
      if (count>1)  // count==1 results in 0
        z_score(buf, count);
      count = 0;
      for (j=0; j<n; j++)
      {
        if (c_matrices[j]->sid == k)
        {
          c_matrices[j]->matrix[i] = buf[count];
          count++;
        }
      }
    }
  }
}

/****************************************
Fisher transform a correlation value (coefficients)
input: the correlation value
output: the transformed value
*****************************************/
float fisherTransformation(float v)
{
  float f1 = 1+v;
  if (f1<=0.0)
  {
    f1 = TINYNUM;
  }
  float f2 = 1-v;
  if (f2<=0.0)
  {
    f2 = TINYNUM;
  }
  return 0.5 * logf(f1/f2);
}

/***************************************
z-score the vectors
input: the vector, the length of the vector
output: write z-scored values to the vector
****************************************/
void z_score(float* v, int n)
{
  int i;
  double mean=0, sd=0;  // float here is not precise enough to handle
  for (i=0; i<n; i++)
  {
    mean += (double)v[i];
    sd += (double)v[i] * v[i]; // double other than float can avoid overflow
  }
  mean /= n;
  //if (sd == n * mean * mean) mean -= 0.1; // to deal with the no variance case
  sd = sd - n * mean * mean;
  if (sd < 0) {cerr<<"sd<0! "<<sd; exit(1);}  // if the type is double above, this won't happen
  sd = sqrt(sd);
  for (i=0; i<n; i++)
  {
    if (sd != 0)
      v[i] = (v[i] - mean) / sd;
    else  // all values are the same
    {
      cerr<<"All values are the same when doing z-score"<<endl;
      exit(1);
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

void MatrixPermutation(RawMatrix** r_matrices, int nSubs)
{
  int row = r_matrices[0]->row;
  int col = r_matrices[0]->col;
  int i, j;
  uint16 buf[row];
  srand(time(NULL));
  ifstream ifile("/state/partition3/yidawang/face_scene/permBook1.txt", ios::in);
  if (!ifile)
  {
		cerr<<"file not found: "<<"/state/partition3/yidawang/face_scene/permBook1.txt"<<endl;
    exit(1);
	}
  int k;
  for (i=0; i<nSubs; i++)
  {
    for (j=0; j<row; j++)
    {
      ifile>>k;
      memcpy((void*)buf, (const void*)&(r_matrices[i]->matrix[j*col]), col*sizeof(uint16));
      memcpy((void*)&(r_matrices[i]->matrix[j*col]), (const void*)&(r_matrices[i]->matrix[k*col]), col*sizeof(double));
      memcpy((void*)&(r_matrices[i]->matrix[k*col]), (const void*)buf, col*sizeof(double));
    }
  }
  ifile.close();
}
