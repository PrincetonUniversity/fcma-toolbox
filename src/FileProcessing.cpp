/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "FileProcessing.h"
#include "common.h"
#include <zlib.h>
#undef __USE_BSD
#include <dirent.h>
#include <sstream>

/*************************
read a bunch of raw matrix files
input: the file directory, the file type e.g. ".nii.gz"
output: an array of raw matrix structs, the number of subjects (reference parameter)
**************************/
RawMatrix** ReadGzDirectory(const char* filepath, const char* filetype, int& nSubs)
{
  DIR *pDir;
  struct dirent *dp;
  char fullfilename[MAXFILENAMELENGTH];
  string strFilenames[MAXSUBJS];  // at most 100 subjects
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
  size_t startPos = fileStr.find_last_of('/');
  size_t endPos = fileStr.find_first_of('.', startPos);
  string sname = fileStr.substr(startPos+1, endPos-startPos-1); // assuming that the subject name doesn't contain '.'
  r_matrix->sname = sname;
  r_matrix->row = row;
  r_matrix->col = col;
  r_matrix->matrix = new float[row*col];
  uint16* data = new uint16[row*col];
  gzread(fp, data, row*col*sizeof(uint16));
  #pragma omp parallel for
  for (int i=0; i<row*col; i++) // not efficient
  {
    r_matrix->matrix[i] = (double)data[i];
  }
  gzclose(fp);
  delete[] data;
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
  size_t startPos = fileStr.find_last_of('/');
  size_t endPos = fileStr.find_first_of('.', startPos);
  string sname = fileStr.substr(startPos+1, endPos-startPos-1); // assuming that the subject name doesn't contain '.'
  r_matrix->sname = sname;
  r_matrix->sid = sid;
  r_matrix->row = nim->nx * nim->ny * nim->nz;
  r_matrix->col = nim->nt;
  r_matrix->nx = nim->nx;
  r_matrix->ny = nim->ny;
  r_matrix->nz = nim->nz;
  r_matrix->matrix = new float[r_matrix->row * r_matrix->col];
  short* data_short=NULL;
  unsigned short* data_ushort=NULL;
  int* data_int=NULL;
  float* data_float=NULL;
  double* data_double=NULL;
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
    case DT_FLOAT64:
      data_double = (double*)nim->data;
      break;
    default:
      cerr<<"wrong data type of data file! "<<nim->datatype<<endl;
      exit(1);
  }
  #pragma omp parallel for
  // the follow if statements don't harm the performance, it's no significant benifit to put the follow ifs to the previous switch
  for (int i=0; i<r_matrix->row*r_matrix->col; i++)
  {
    int t1 = i / r_matrix->row;
    int t2 = i % r_matrix->row;
    if (data_short!=NULL)
    {
      r_matrix->matrix[t2*r_matrix->col+t1] = (float)data_short[i];  // transpose, because data is time (row) * voxel (column), r_matrix wants voxel (row) * time (column)
    }
    if (data_ushort!=NULL)
    {
      r_matrix->matrix[t2*r_matrix->col+t1] = (float)data_ushort[i];  // transpose, because data is time (row) * voxel (column), r_matrix wants voxel (row) * time (column)
    }
    if (data_int!=NULL)
    {
      r_matrix->matrix[t2*r_matrix->col+t1] = (float)data_int[i];  // transpose, because data is time (row) * voxel (column), r_matrix wants voxel (row) * time (column)
    }
    if (data_float!=NULL)
    {
      r_matrix->matrix[t2*r_matrix->col+t1] = (float)data_float[i];  // transpose, because data is time (row) * voxel (column), r_matrix wants voxel (row) * time (column)
    }
    if (data_double!=NULL)
    {
      r_matrix->matrix[t2*r_matrix->col+t1] = (float)data_double[i];  // transpose, because data is time (row) * voxel (column), r_matrix wants voxel (row) * time (column)
    }
  }
  nifti_image_free(nim);
  return r_matrix;
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
  double* data_double = NULL;
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
    case DT_FLOAT64:
      data_double = (double*)nim->data;
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
    masked_matrix->matrix = new float[row*col];
    float* src_mat = r_matrices[i]->matrix;
    float* dest_mat = masked_matrix->matrix;
    int count = 0;
    for (j=0; j<row; j++)
    {
      if (data_int!=NULL && data_int[j])
      {
        memcpy(&(dest_mat[count*col]), &(src_mat[j*col]), col*sizeof(float));
        count++;
      }
      if (data_short!=NULL && data_short[j])
      {
        memcpy(&(dest_mat[count*col]), &(src_mat[j*col]), col*sizeof(float));
        count++;
      }
      if (data_uchar!=NULL && data_uchar[j])
      {
        memcpy(&(dest_mat[count*col]), &(src_mat[j*col]), col*sizeof(float));
        count++;
      }
      if (data_float!=NULL && data_float[j]>=1-TINYNUM)
      {
        memcpy(&(dest_mat[count*col]), &(src_mat[j*col]), col*sizeof(float));
        count++;
      }
      if (data_double!=NULL && data_double[j]>=1-TINYNUM)
      {
        memcpy(&(dest_mat[count*col]), &(src_mat[j*col]), col*sizeof(float));
        count++;
      }
    }
    masked_matrix->row = count; // update the row information
    masked_matrices[i] = masked_matrix;
  }
  nifti_image_free(nim);
  return masked_matrices;
}

/************************************************
Generate a masked matrix using the mask file
input: the raw data matrix, the mask file
output: a masked matrix
*************************************************/
RawMatrix* GetMaskedMatrix(RawMatrix* r_matrix, const char* maskFile)
{
  RawMatrix* masked_matrix = new RawMatrix;
  nifti_image* nim = nifti_image_read(maskFile, 1);
  int j;
  if (nim == NULL)
  {
    cerr<<"file not found: "<<maskFile<<endl;
    exit(1);
  }
  int* data_int = NULL;
  short* data_short = NULL;
  unsigned char* data_uchar = NULL;
  float* data_float = NULL;
  double* data_double = NULL;
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
    case DT_FLOAT64:
      data_double = (double*)nim->data;
      break;
    default:
      cerr<<"wrong data type of mask file!"<<endl;
      exit(1);
  }
  masked_matrix->sid = r_matrix->sid;
  masked_matrix->row = r_matrix->row;
  masked_matrix->col = r_matrix->col;
  masked_matrix->nx = r_matrix->nx;
  masked_matrix->ny = r_matrix->ny;
  masked_matrix->nz = r_matrix->nz;
  int row = r_matrix->row;
  int col = r_matrix->col;
  masked_matrix->matrix = new float[row*col];
  float* src_mat = r_matrix->matrix;
  float* dest_mat = masked_matrix->matrix;
  int count = 0;
  for (j=0; j<row; j++)
  {
    if (data_int!=NULL && data_int[j])
    {
      memcpy(&(dest_mat[count*col]), &(src_mat[j*col]), col*sizeof(float));
      count++;
    }
    if (data_short!=NULL && data_short[j])
    {
      memcpy(&(dest_mat[count*col]), &(src_mat[j*col]), col*sizeof(float));
      count++;
    }
    if (data_uchar!=NULL && data_uchar[j])
    {
      memcpy(&(dest_mat[count*col]), &(src_mat[j*col]), col*sizeof(float));
      count++;
    }
    if (data_float!=NULL && data_float[j]>=1-TINYNUM)
    {
      memcpy(&(dest_mat[count*col]), &(src_mat[j*col]), col*sizeof(float));
      count++;
    }
    if (data_double!=NULL && data_double[j]>=1-TINYNUM)
    {
      memcpy(&(dest_mat[count*col]), &(src_mat[j*col]), col*sizeof(float));
      count++;
    }
    masked_matrix->row = count; // update the row information
  }
  nifti_image_free(nim);
  return masked_matrix;
}

/************************************************
Generate masked point location using the mask file
input: the raw point location, the number of voxels after masking, the mask file
output: a masked point location array
*************************************************/
Point* GetMaskedPts(Point* pts, int nMaskedVoxels, const char* maskFile)
{
  Point* masked_pts = new Point[nMaskedVoxels];
  nifti_image* nim = nifti_image_read(maskFile, 1);
  int i;
  if (nim == NULL)
  {
    cerr<<"file not found: "<<maskFile<<endl;
    exit(1);
  }
  int* data_int = NULL;
  short* data_short = NULL;
  unsigned char* data_uchar = NULL;
  float* data_float = NULL;
  double* data_double = NULL;
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
    case DT_FLOAT64:
      data_double = (double*)nim->data;
      break;
    default:
      cerr<<"wrong data type of mask file!"<<endl;
      exit(1);
  }
  int nTotalVoxels = nim->nx*nim->ny*nim->nz;
  int count=0;
  for (i=0; i<nTotalVoxels; i++) // get only the masked voxels
  {
    if (data_int!=NULL && data_int[i])
    {
      masked_pts[count].x=pts[i].x;
      masked_pts[count].y=pts[i].y;
      masked_pts[count].z=pts[i].z;
      count++;
    }
    if (data_short!=NULL && data_short[i])
    {
      masked_pts[count].x=pts[i].x;
      masked_pts[count].y=pts[i].y;
      masked_pts[count].z=pts[i].z;
      count++;
    }
    if (data_uchar!=NULL && data_uchar[i])
    {
      masked_pts[count].x=pts[i].x;
      masked_pts[count].y=pts[i].y;
      masked_pts[count].z=pts[i].z;
      count++;
    }
    if (data_float!=NULL && data_float[i]>=1-TINYNUM)
    {
      masked_pts[count].x=pts[i].x;
      masked_pts[count].y=pts[i].y;
      masked_pts[count].z=pts[i].z;
      count++;
    }
    if (data_double!=NULL && data_double[i]>=1-TINYNUM)
    {
      masked_pts[count].x=pts[i].x;
      masked_pts[count].y=pts[i].y;
      masked_pts[count].z=pts[i].z;
      count++;
    }
  }
  nifti_image_free(nim);
  return masked_pts;
}

/******************************
Generate trial information for same trials across subjects
input: the number of subjects, the shift number, the number of trials, the block information file
output: the trial data structure array, the number of trials
*******************************/
Trial* GenRegularTrials(int nSubs, int nShift, int& nTrials, const char* file)
{
  ifstream ifile(file);
  if (!ifile)
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
      trials[index].tid_withinsubj = j;
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
  Trial* trials = new Trial[nSubs*12];  //maximal 12 trials per subject
  int index=0;
  for (i=0; i<nSubs; i++)
  {
    string blockFileStr = r_matrices[i]->sname + ".txt";
    blockFileStr = dirStr + blockFileStr;
    ifstream ifile(blockFileStr.c_str());
    if (!ifile)
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
      trials[index].tid_withinsubj = j;
      trials[index].tid = j;
      index++;
    }
    ifile.close();
    nTrials += nPerSubs;
  }
  return trials;
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

/*******************************
write data to a compressed nifti file
input: the nifti data file name (with or without extension), the sample nifti file to refer to, the data, the data type(defined in nifti1.h)
output: write the data to the file
********************************/
void WriteNiiGzData(const char* outputFile, const char* refFile, void* data, int dataType)
{
  nifti_image* nim = nifti_image_read(refFile, 1); // 1 means reading the data as well
  if (nim == NULL)
  {
    cerr<<"sample file not found: "<<refFile<<endl;
    exit(1);
  }
  nifti_image* nim2 = nifti_copy_nim_info(nim);
  char* newFileName = nifti_makeimgname((char*)outputFile, nim->nifti_type, 0, 1);  //3rd argument: 0 means overwrite the existing file, 1 means returning error if the file exists; 4th argument: 0 means not compressed, 1 means compressed
  delete nim2->fname; // delete the old filename to avoid memory leak
  nim2->fname = newFileName;
  nim2->datatype = dataType;
  nim2->nbyper = getSizeByDataType(dataType);
  nim2->data = data;
  nifti_image_write(nim2);
  nifti_image_free(nim);
  nifti_image_free(nim2);
  return;
}

/*******************************
write 4D data to a compressed nifti file
input: the nifti data file name (with or without extension), the sample nifti file to refer to, the data, the data type(defined in nifti1.h), the time (4th) dimension
output: write the data to the file
********************************/
void Write4DNiiGzData(const char* outputFile, const char* refFile, void* data, int dataType, int nt)
{
  nifti_image* nim = nifti_image_read(refFile, 1); // 1 means reading the data as well
  if (nim == NULL)
  {
    cerr<<"sample file not found: "<<refFile<<endl;
    exit(1);
  }
  nifti_image* nim2 = nifti_copy_nim_info(nim);
  char* newFileName = nifti_makeimgname((char*)outputFile, nim->nifti_type, 0, 1);  //3rd argument: 0 means overwrite the existing file, 1 means returning error if the file exists; 4th argument: 0 means not compressed, 1 means compressed
  delete nim2->fname; // delete the old filename to avoid memory leak
  nim2->fname = newFileName;
  nim2->datatype = dataType;
  nim2->nbyper = getSizeByDataType(dataType);
  nim2->nt = nt;
  nim2->dim[0] = 4;
  nim2->dim[4] = nt;
  nim2->nvox = nim2->nx*nim2->ny*nim2->nz*nim2->nt;
  nim2->data = data;
  nifti_image_write(nim2);
  nifti_image_free(nim);
  nifti_image_free(nim2);
  return;
}

/*******************************
generate data to be written to a nifti file based on a mask file, the input data is specifically from VoxelScore struct
input: the mask nifti file name, the data, the length of the data array, the data type(defined in nifti1.h, DT_SIGNED_INT indicates to generate voxel id file, DT_FLOAT indicates to generate voxel score file)
output: the data that is ready to be written to a nifti file
********************************/
void* GenerateNiiDataFromMask(const char* maskFile, VoxelScore* scores, int length, int dataType)
{
  nifti_image* nim = nifti_image_read(maskFile, 1); // 1 means reading the data as well
  if (nim == NULL)
  {
    cerr<<"mask file not found: "<<maskFile<<" in GenerateNiiDataFromMask"<<endl;
    exit(1);
  }
  int* data_int = NULL;
  short* data_short = NULL;
  unsigned char* data_uchar = NULL;
  float* data_float = NULL;
  double* data_double = NULL;
  switch (nim->datatype)
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
    case DT_FLOAT:
      data_float = (float*)nim->data;
      break;
    case DT_FLOAT64:
      data_double = (double*)nim->data;
      break;
    default:
      cerr<<"wrong data type of mask file!"<<" in GenerateNiiDataFromMask"<<endl;
      exit(1);
  }
  int nVoxels = nim->nx*nim->ny*nim->nz;
  void* returnData = NULL;
  int* returnData_int = NULL;
  float* returnData_float = NULL;
  if (dataType == DT_SIGNED_INT)
  {
    returnData_int = new int[nVoxels];
    memset(returnData_int, 0, nVoxels*sizeof(int));
    returnData = (void*)returnData_int;
  }
  else if (dataType == DT_FLOAT)
  {
    returnData_float = new float[nVoxels];
    memset(returnData_float, 0, nVoxels*sizeof(float));
    returnData = (void*)returnData_float;
  }
  else
  {
    cerr<<"wrong data type of return data!"<<" in GenerateNiiDataFromMask"<<endl;
    exit(1);
  }
  // generate data from scores
  void* maskedData=new int[length];  // int and float have the same data size
  if (dataType == DT_SIGNED_INT)
  {
    #pragma omp parallel for
    for (int i=0; i<length; i++)
    {
      ((int*)maskedData)[scores[i].vid] = i+1; //voxel list is 0-based
    }
  }
  else  // must be DT_FLOAT here
  {
    #pragma omp parallel for
    for (int i=0; i<length; i++)
    {
      ((float*)maskedData)[scores[i].vid] = scores[i].score; //voxel list is 0-based
    }
  }
  //generate data done

  int count=0;
  // because of the count variable, no OMP here
  for (int i=0; i<nVoxels; i++)
  {
    if (data_int!=NULL && data_int[i])
    {
      if (count==length)
      {
        cerr<<"number of scores is larger than number of masked voxels "<<length<<"!"<<" in GenerateNiiDataFromMask"<<endl;
        exit(1);
      }
      if (dataType == DT_SIGNED_INT)
      {
        returnData_int[i] = ((int*)maskedData)[count];
      }
      else  // must be DT_FLOAT here
      {
        returnData_float[i] = ((float*)maskedData)[count];
      }
      count++;
    }
    if (data_short!=NULL && data_short[i])
    {
      if (count==length)
      {
        cerr<<"number of scores is larger than number of masked voxels "<<length<<"!"<<" in GenerateNiiDataFromMask"<<endl;
        exit(1);
      }
      if (dataType == DT_SIGNED_INT)
      {
        returnData_int[i] = ((int*)maskedData)[count];
      }
      else  // must be DT_FLOAT here
      {
        returnData_float[i] = ((float*)maskedData)[count];
      }
      count++;
    }
    if (data_uchar!=NULL && data_uchar[i])
    {
      if (count==length)
      {
        cerr<<"number of scores is larger than number of masked voxels "<<length<<"!"<<" in GenerateNiiDataFromMask"<<endl;
        exit(1);
      }
      if (dataType == DT_SIGNED_INT)
      {
        returnData_int[i] = ((int*)maskedData)[count];
      }
      else  // must be DT_FLOAT here
      {
        returnData_float[i] = ((float*)maskedData)[count];
      }
      count++;
    }
    if (data_float!=NULL && data_float[i]>=1-TINYNUM)
    {
      if (count==length)
      {
        cerr<<"number of scores is larger than number of masked voxels "<<length<<"!"<<" in GenerateNiiDataFromMask"<<endl;
        exit(1);
      }
      if (dataType == DT_SIGNED_INT)
      {
        returnData_int[i] = ((int*)maskedData)[count];
      }
      else  // must be DT_FLOAT here
      {
        returnData_float[i] = ((float*)maskedData)[count];
      }
      count++;
    }
    if (data_double!=NULL && data_double[i]>=1-TINYNUM)
    {
      if (count==length)
      {
        cerr<<"number of scores is larger than number of masked voxels "<<length<<"!"<<" in GenerateNiiDataFromMask"<<endl;
        exit(1);
      }
      if (dataType == DT_SIGNED_INT)
      {
        returnData_int[i] = ((int*)maskedData)[count];
      }
      else  // must be DT_FLOAT here
      {
        returnData_float[i] = ((float*)maskedData)[count];
      }
      count++;
    }
  }
  nifti_image_free(nim);
  return returnData;
}

/*************************************
take a nifti datatype as input
return its size in bytes to feed nbyter of nifti_image
**************************************/
inline int getSizeByDataType(int datatype)
{
  int nbyter =-1;
  switch (datatype)
  {
    case 2: // DT_UNSIGNED_CHAR, DT_UINT8
    case 256: // DT_INT8
      nbyter = 1; // 1 byte
      break;
    case 4: // DT_SIGNED_SHORT, DT_INT16
    case 512: // DT_UINT16
      nbyter = 2; // 2 bytes
      break;
    case 8: // DT_SIGNED_INT, DT_INT32
    case 768: // DT_UINT32
    case 16:  //  DT_FLOAT, DT_FLOAT32
      nbyter = 4; // 4 bytes
      break;
    case 64:  //DT_DOUBLE, DT_FLOAT64
      nbyter = 8;
      break;
    default:
      cerr<<"wrong data type of nifti file in getSizeByDataType!"<<endl;
      exit(1);
  }
  return nbyter;
}
