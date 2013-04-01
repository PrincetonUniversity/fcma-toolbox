#include "Preprocessing.h"
#include "common.h"
#include <zlib.h>
#undef __USE_BSD
#include <dirent.h>
#include <sstream>
#include <nifti1_io.h>

/*************************
read a bunch of raw matrix files
input: the file directory, the file type e.g. ".nii.gz"
output: an array of raw matrix structs, the number of subjects (reference parameter)
**************************/
RawMatrix** ReadGzDirectory(const char* filepath, const char* filetype, int& nSubs)
{
  DIR *pDir;
  struct dirent *dp;
  char fullfilename[100];
  string strFilenames[100];  // at most 100 subjects
  if ((pDir=opendir(filepath)) == NULL)
  {
    cerr<<"invalid directory"<<endl;
    exit(1);
  }
  nSubs = 0;
  while ((dp=readdir(pDir)) != NULL)
  {
    if ((strstr(dp->d_name, filetype)) != NULL)
    {
      nSubs++;
    }
  }
  closedir(pDir);
  if ((pDir=opendir(filepath)) == NULL)
  {
    cerr<<"invalid directory: "<<filepath<<endl;
    exit(1);
  }
  RawMatrix** r_matrices = new RawMatrix*[nSubs];
  int count = 0;
  while ((dp=readdir(pDir)) != NULL)
  {
    if ((strstr(dp->d_name, filetype)) != NULL)
    {
      sprintf(fullfilename, "%s", filepath);
      strcat(fullfilename, dp->d_name);
      strFilenames[count] = string(fullfilename);
      count++;
    }
  }
  sort(strFilenames, strFilenames+count);
  for (int i=0; i<count; i++)
  {
    //cout<<strFilenames[i]<<endl; // output the file name to know the file sequance
    //r_matrices[i] = ReadGzData(strFilenames[i], i);
    r_matrices[i] = ReadNiiGzData(strFilenames[i], i);
  }
  closedir(pDir);
  return r_matrices;
}


/*************************
read matrix data from gz files, the first eight bytes are two 32-bit ints, indicating the row and col of the following matrix
input: the gz file name, subject id
output: the raw matrix data
**************************/
RawMatrix* ReadGzData(string fileStr, int sid)
{
  const char* file = fileStr.c_str();
  gzFile fp = gzopen(file, "rb");
  //cout<<sid+1<<": "<<file<<endl;
  if (fp == NULL)
  {
    cout<<"file not found: "<<file<<endl;
    exit(1);
  }
  RawMatrix* r_matrix = new RawMatrix();
  r_matrix->sid = sid;
  int row, col;
  gzread(fp, &row, sizeof(int));
  gzread(fp, &col, sizeof(int));
  int startPos = fileStr.find_last_of('/');
  int endPos = fileStr.find_first_of('.', startPos);
  string sname = fileStr.substr(startPos+1, endPos-startPos-1); // assuming that the subject name doesn't contain '.'
  r_matrix->sname = sname;
  r_matrix->row = row;
  r_matrix->col = col;
  r_matrix->matrix = new double[row*col];
  uint16* data = new uint16[row*col];
  gzread(fp, data, row*col*sizeof(uint16));
  for (int i=0; i<row*col; i++) // not efficient
  {
    r_matrix->matrix[i] = (double)data[i];  // transpose, because data is time (row) * voxel (column), r_matrix wants voxel (row) * time (column)
  }
  gzclose(fp);
  delete data;
  return r_matrix;
}

/*************************
read matrix data from nii.gz files
input: the gz file name, subject id
output: the raw matrix data
**************************/
RawMatrix* ReadNiiGzData(string fileStr, int sid)
{
  const char* file = fileStr.c_str();
  nifti_image* nim;
  nim = nifti_image_read(file, 1);
  if (nim == NULL)
  {
    cerr<<"file not found: "<<file<<endl;
    exit(1);
  }
  /*cout<<"fname: "<<nim->fname<<endl;
  cout<<"iname: "<<nim->iname<<endl;
  cout<<"nifti_type: "<<nim->nifti_type<<endl;
  cout<<"dx: "<<nim->dx<<endl;
  cout<<"dy: "<<nim->dy<<endl;
  cout<<"dz: "<<nim->dz<<endl;
  cout<<"byteorder: "<<nim->byteorder<<endl;
  exit(1);*/
  RawMatrix* r_matrix = new RawMatrix();
  int startPos = fileStr.find_last_of('/');
  int endPos = fileStr.find_first_of('.', startPos);
  string sname = fileStr.substr(startPos+1, endPos-startPos-1); // assuming that the subject name doesn't contain '.'
  r_matrix->sname = sname;
  r_matrix->sid = sid;
  r_matrix->row = nim->nx * nim->ny * nim->nz;
  r_matrix->col = nim->nt;
  r_matrix->nx = nim->nx;
  r_matrix->ny = nim->ny;
  r_matrix->nz = nim->nz;
  r_matrix->matrix = new double[r_matrix->row * r_matrix->col];
  short* data_short=NULL;
  unsigned short* data_ushort=NULL;
  int* data_int=NULL;
  float* data_float=NULL;
  switch (nim->datatype)  // now only get one type
  {
    case DT_SIGNED_SHORT:
      data_short = (short*)nim->data;
      break;
    case DT_UINT16:
      data_ushort = (unsigned short*)nim->data;
      break;
    case DT_SIGNED_INT:
      data_int = (int*)nim->data;
      break;
    case DT_FLOAT:
      data_float = (float*)nim->data;
      break;
    default:
      cerr<<"wrong data type of data file! "<<nim->datatype<<endl;
      exit(1);
  }
  for (int i=0; i<r_matrix->row*r_matrix->col; i++) // not efficient
  {
    int t1 = i / r_matrix->row;
    int t2 = i % r_matrix->row;
    if (data_short!=NULL)
    {
      r_matrix->matrix[t2*r_matrix->col+t1] = (double)data_short[i];  // transpose, because data is time (row) * voxel (column), r_matrix wants voxel (row) * time (column)
    }
    if (data_ushort!=NULL)
    {
      r_matrix->matrix[t2*r_matrix->col+t1] = (double)data_ushort[i];  // transpose, because data is time (row) * voxel (column), r_matrix wants voxel (row) * time (column)
    }
    if (data_int!=NULL)
    {
      r_matrix->matrix[t2*r_matrix->col+t1] = (double)data_int[i];  // transpose, because data is time (row) * voxel (column), r_matrix wants voxel (row) * time (column)
    }
    if (data_float!=NULL)
    {
      r_matrix->matrix[t2*r_matrix->col+t1] = (double)data_float[i];  // transpose, because data is time (row) * voxel (column), r_matrix wants voxel (row) * time (column)
    }
  }
  nifti_image_free(nim);
  return r_matrix;
}

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
    double* mat = r_matrices[i]->matrix;
    for (j=0; j<row; j++)
    {
      bool flag = true;
      for (k=0; k<col; k++)
      {
        flag &= (mat[j*col+k]<=2.0);  // 2 is a threshold for "almost" all-zero
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
  ofile.close();
  exit(1);*/
  for (i=0; i<nSubs; i++) // remove the all-zero voxels
  {
    int col = r_matrices[i]->col;
    double* mat = r_matrices[i]->matrix;
    count = 0;
    for (j=0; j<row; j++)
    {
      if (flags[j])
      {
        memcpy(&(mat[count*col]), &(mat[j*col]), col*sizeof(double));
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
    double* mat = r_matrices[i]->matrix;
    count = 0;
    for (j=0; j<row; j++)
    {
      if (data[j])
      {
        memcpy(&(mat[count*col]), &(mat[j*col]), col*sizeof(double));
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

/************************************************
Generate masked matrices using the mask file
input: the raw data array, the number of subjects, the mask file
output: a masked data array
*************************************************/
RawMatrix** GetMaskedMatrices(RawMatrix** r_matrices, int nSubs, const char* maskFile)
{
  RawMatrix** masked_matrices = new RawMatrix*[nSubs];
  nifti_image* nim = nifti_image_read(maskFile, 1);
  int i, j;
  if (nim == NULL)
  {
    cerr<<"file not found: "<<maskFile<<endl;
    exit(1);
  }
  int* data_int = NULL;
  short* data_short = NULL;
  unsigned char* data_uchar = NULL;
  float* data_float = NULL;
  //cout<<nim->nx<<" "<<nim->ny<<" "<<nim->nz<<endl; exit(1);
  switch (nim->datatype)  // now only get one type
  {
    case DT_SIGNED_INT:
      data_int = (int*)nim->data;
      break;
    case DT_SIGNED_SHORT:
      data_short = (short*)nim->data;
      break;
    case DT_UNSIGNED_CHAR:
      data_uchar = (unsigned char*)nim->data;
      break;
    case DT_FLOAT32:
      data_float = (float*)nim->data;
      break;
    default:
      cerr<<"wrong data type of mask file!"<<endl;
      exit(1);
  }
  for (i=0; i<nSubs; i++) // get only the masked voxels
  {
    RawMatrix* masked_matrix = new RawMatrix();
    masked_matrix->sid = r_matrices[i]->sid;
    masked_matrix->row = r_matrices[i]->row;
    masked_matrix->col = r_matrices[i]->col;
    masked_matrix->nx = r_matrices[i]->nx;
    masked_matrix->ny = r_matrices[i]->ny;
    masked_matrix->nz = r_matrices[i]->nz;
    int row = r_matrices[i]->row;
    int col = r_matrices[i]->col;
    masked_matrix->matrix = new double[row*col];
    double* src_mat = r_matrices[i]->matrix;
    double* dest_mat = masked_matrix->matrix;
    int count = 0;
    for (j=0; j<row; j++)
    {
      if (data_int!=NULL && data_int[j])
      {
        memcpy(&(dest_mat[count*col]), &(src_mat[j*col]), col*sizeof(double));
        count++;
      }
      if (data_short!=NULL && data_short[j])
      {
        memcpy(&(dest_mat[count*col]), &(src_mat[j*col]), col*sizeof(double));
        count++;
      }
      if (data_uchar!=NULL && data_uchar[j])
      {
        memcpy(&(dest_mat[count*col]), &(src_mat[j*col]), col*sizeof(double));
        count++;
      }
      if (data_float!=NULL && data_float[j]>=1-TINYNUM)
      {
        memcpy(&(dest_mat[count*col]), &(src_mat[j*col]), col*sizeof(double));
        count++;
      }
    }
    masked_matrix->row = count; // update the row information
    masked_matrices[i] = masked_matrix;
  }
  nifti_image_free(nim);
  return masked_matrices;
}

/******************************
Generate trial information for same trials across subjects
input: the number of subjects, the shift number, the number of trials, the block information file
output: the trial data structure array, the number of trials
*******************************/
Trial* GenRegularTrials(int nSubs, int nShift, int& nTrials, const char* file)
{
  ifstream ifile(file);
  if (ifile==NULL)
  {
    cerr<<"no block file found!"<<endl;
    exit(1);
  }
  int nPerSubs = -1;
  ifile>>nPerSubs;
  Trial* trials = new Trial[nSubs*nPerSubs];  //12 trials per subject
  int trial_labels[nPerSubs];// = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int scs[nPerSubs];// = {4, 24, 44, 64, 84, 104, 124, 144, 164, 184, 204, 224};
  int ecs[nPerSubs];// = {15, 35, 55, 75, 95, 115, 135, 155, 175, 195, 215, 235};
  //int trial_labels[6] = {0, 1, 0, 1, 0, 1};
  //int scs[6] = {4, 24, 84, 104, 164, 184};
  //int ecs[6] = {15, 35, 95, 115, 175, 195};
  int i, j;
  for (i=0; i<nPerSubs; i++)
  {
    ifile>>trial_labels[i]>>scs[i]>>ecs[i];
  }
  ifile.close();
  for (i=0; i<nSubs; i++)
  {
    for (j=0; j<nPerSubs; j++)
    {
      int index = i*nPerSubs+j;
      trials[index].sid = i;
      trials[index].label = trial_labels[j];
      trials[index].sc = scs[j] + nShift;
      trials[index].ec = ecs[j] + nShift;
      if (i<nSubs/2)
      {
        trials[index].tid = j;
      }
      else
      {
        if (j % 2 == 0)
        {
          trials[index].tid = j + 1;
        }
        else
        {
          trials[index].tid = j - 1;
        }
      }
    }
  }
  nTrials = nSubs * nPerSubs;
  return trials;
}

/******************************
Generate block information from a block directory, one-to-one mapping to the data file.
The block files are in txt extension
input: the number of subjects, the shift number, the number of trials, the block information directory
output: the trial data structure array, the number of trials
*******************************/
Trial* GenBlocksFromDir(int nSubs, int nShift, int& nTrials, RawMatrix** r_matrices, const char* dir)
{
  DIR *pDir;
  if ((pDir=opendir(dir)) == NULL)
  {
    cerr<<"invalid block information directory"<<endl;
    exit(1);
  }
  closedir(pDir);
  string dirStr = string(dir);
  if (dirStr[dirStr.length()-1] != '/')
  {
    dirStr += '/';
  }
  int i, j;
  nTrials = 0;
  Trial* trials = new Trial[nSubs*12];  //12 trials per subject
  int index=0;
  for (i=0; i<nSubs; i++)
  {
    string blockFileStr = r_matrices[i]->sname + ".txt";
    blockFileStr = dirStr + blockFileStr;
    ifstream ifile(blockFileStr.c_str());
    if (ifile==NULL)
    {
      cerr<<"no block file found!"<<endl;
      exit(1);
    }
    int nPerSubs = -1;
    ifile>>nPerSubs;
    int trial_labels[nPerSubs];
    int scs[nPerSubs];
    int ecs[nPerSubs];
    for (j=0; j<nPerSubs; j++)
    {
      ifile>>trial_labels[j]>>scs[j]>>ecs[j];
      trials[index].sid = i;
      trials[index].label = trial_labels[j];
      trials[index].sc = scs[j] + nShift;
      trials[index].ec = ecs[j] + nShift;
      trials[index].tid = j;
      index++;
    }
    ifile.close();
    nTrials += nPerSubs;
  }
  return trials;
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
    double buf[n];
    int j, k;
    // need to do RT regression and get z-scored subject by subject
    for (k=0; k<nSubs; k++)
    {
      int count = 0;
      for (j=0; j<n; j++)
      {
        if (c_matrices[j]->sid == k)
        {
          buf[count] = double(c_matrices[j]->matrix[i]);
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
        buf[j] = fisherTransformation(buf[j]);
      z_score(buf, count);
      count = 0;
      for (j=0; j<n; j++)
      {
        if (c_matrices[j]->sid == k)
        {
          c_matrices[j]->matrix[i] = float(buf[count]);
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
void z_score(double* v, int n)
{
  int i;
  double mean=0, sd=0;  // float here is not precise enough to handle
  for (i=0; i<n; i++)
  {
    mean += v[i];
    sd += v[i] * v[i]; // double other than float can avoid overflow
  }
  mean /= n;
  //if (sd == n * mean * mean) mean -= 0.1; // to deal with the no variance case
  sd = sd - n * mean * mean;
  if (sd < 0) {cerr<<"sd<0! "<<sd; exit(1);}
  sd = sqrt(sd);
  for (i=0; i<n; i++)
  {
    if (sd != 0)
      v[i] = (v[i] - mean) / sd;
    else
      v[i] = 0;
  }
}

/***************************
read the location information, from a binary file, the file has row (#voxels) and column (#TRs) in the beginning, followed by every voxel's 3D location
input: the file name
output: the location struct
****************************/
Point* ReadLocInfo(const char* file)
{
  int row, col;
  FILE* fp = fopen(file, "rb");
  fread((void*)&row, sizeof(int), 1, fp);
  fread((void*)&col, sizeof(int), 1, fp);
  Point* pts = new Point[row];
  fread((void*)pts, sizeof(Point), row, fp);
  fclose(fp);
  // one time adjustment
  /*int i;
  for (i=0; i<row; i++)
  {
    pts[i].x -= 32;
    pts[i].y -= 32;
    pts[i].z -= 17;
  }*/
  return pts;
}

/***************************
generrate the initial location information following the nifti format, using the nx ny nz information getting from the data nifti files
input: an arbitrary raw matrix, assuming that all subjects have the same format
output: the location struct
****************************/
Point* ReadLocInfoFromNii(RawMatrix* r_matrix)
{
  int row = r_matrix->row;
  int nx = r_matrix->nx;
  int ny = r_matrix->ny;
  int nz = r_matrix->nz;
  Point* pts = new Point[row];
  for (int z=0; z<nz; z++)
    for (int y=0; y<ny; y++)
      for (int x=0; x<nx; x++)
      {
        pts[z*ny*nx+y*nx+x].x = x;
        pts[z*ny*nx+y*nx+x].y = y;
        pts[z*ny*nx+y*nx+x].z = z;
      }
  return pts;
}

/*******************************
read the RT matrices information
input: the file name
output: the RT matrix array, the number of matrices (in the inputting parameter)
********************************/
double** ReadRTMatrices(const char* file, int& nSubs)
{
  int nBlocksPerSub = -1;
  FILE* fp = fopen(file, "rb");
  fread((void*)&nSubs, sizeof(int), 1, fp);
  fread((void*)&nBlocksPerSub, sizeof(int), 1, fp);
  double** rtMatrices = new double*[nSubs];
  int i;
  for (i=0; i<nSubs; i++)
  {
    double* rtMatrix = new double[nBlocksPerSub*nBlocksPerSub];
    fread((void*)rtMatrix, sizeof(double), nBlocksPerSub*nBlocksPerSub, fp);
    rtMatrices[i] = rtMatrix;
  }
  return rtMatrices;
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
    avg_matrices[i]->matrix = new double[row*nTrialsPerSub];
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
