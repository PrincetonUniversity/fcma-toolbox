/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/
#ifndef __MIC__
#include "FileProcessing.h"
#include "common.h"
#include <zlib.h>
#undef __USE_BSD
#include <dirent.h>
#include <cassert>
#include <sstream>
#include "ErrorHandling.h"
#include <hdf5.h>
#include <hdf5_hl.h>

/*************************
read a bunch of raw matrix files
input: the file directory, the file type e.g. ".nii.gz"
output: an array of raw matrix structs, the number of subjects (reference
parameter)
**************************/
RawMatrix** ReadGzDirectory(const char* filepath, const char* filetype,
                            int& nSubs) {
  DIR* pDir;
  struct dirent* dp;
  char fullfilename[MAXFILENAMELENGTH];
  std::string strFilenames[MAXSUBJS];
  if ((pDir = opendir(filepath)) == NULL) {
    FATAL("invalid directory");
  }
  nSubs = 0;
  while ((dp = readdir(pDir)) != NULL) {
    if ((strstr(dp->d_name, filetype)) != NULL) {
      nSubs++;
    }
  }
  closedir(pDir);
  if ((pDir = opendir(filepath)) == NULL) {
    FATAL("invalid directory: " << filepath);
  }
  RawMatrix** r_matrices = new RawMatrix* [nSubs];
  int count = 0;
  while ((dp = readdir(pDir)) != NULL) {
    if ((strstr(dp->d_name, filetype)) != NULL) {
      sprintf(fullfilename, "%s", filepath);
      strcat(fullfilename, dp->d_name);
      strFilenames[count] = std::string(fullfilename);
      count++;
    }
  }
  sort(strFilenames, strFilenames + count);
  for (int i = 0; i < count; i++) {
    // cout<<"file "<<i<<": "<<strFilenames[i]<<endl; // output the file name to
    // know the file sequance
    // r_matrices[i] = ReadGzData(strFilenames[i], i);
    r_matrices[i] = ReadNiiGzData(strFilenames[i], i);
  }
  closedir(pDir);
  return r_matrices;
}

/*************************
read matrix data from gz files, the first eight bytes are two 32-bit ints,
indicating the row and col of the following matrix
input: the gz file name, subject id
output: the raw matrix data
**************************/
RawMatrix* ReadGzData(std::string fileStr, int sid) {
  const char* file = fileStr.c_str();
  gzFile fp = gzopen(file, "rb");
  // cout<<sid+1<<": "<<file<<endl;
  if (fp == NULL) {
    FATAL("file not found: " << file);
  }
  RawMatrix* r_matrix = new RawMatrix();
  r_matrix->sid = sid;
  int row, col;
  gzread(fp, &row, sizeof(int));
  gzread(fp, &col, sizeof(int));
  size_t startPos = fileStr.find_last_of('/');
  size_t endPos = fileStr.find_first_of('.', startPos);
  std::string sname = fileStr.substr(
      startPos + 1, endPos - startPos - 1);  // assuming that the subject name
                                             // doesn't contain '.'
  r_matrix->sname = sname;
  r_matrix->row = row;
  r_matrix->col = col;
  r_matrix->matrix = new float[row * col];
  uint16* data = new uint16[row * col];
  gzread(fp, data, row * col * sizeof(uint16));
#pragma omp parallel for
  for (int i = 0; i < row * col; i++)  // not efficient
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
RawMatrix* ReadNiiGzData(std::string fileStr, int sid) {
  const char* file = fileStr.c_str();
  nifti_image* nim;
  nim = nifti_image_read(file, 1);
  if (nim == NULL) {
    FATAL("file not found: " << file);
  }
  assert(nim);
  /*cout<<"fname: "<<nim->fname<<endl;
  cout<<"iname: "<<nim->iname<<endl;
  cout<<"nifti_type: "<<nim->nifti_type<<endl;
  cout<<"dx: "<<nim->dx<<endl;
  cout<<"dy: "<<nim->dy<<endl;
  cout<<"dz: "<<nim->dz<<endl;
  cout<<"byteorder: "<<nim->byteorder<<endl;
  FATAL("just debugging");*/
  RawMatrix* r_matrix = new RawMatrix();
  size_t startPos = fileStr.find_last_of('/');
  size_t endPos = fileStr.find_first_of('.', startPos);
  std::string sname = fileStr.substr(
      startPos + 1, endPos - startPos - 1);  // assuming that the subject name
                                             // doesn't contain '.'
  r_matrix->sname = sname;
  r_matrix->sid = sid;
  r_matrix->row = nim->nx * nim->ny * nim->nz;
  r_matrix->col = nim->nt;
  r_matrix->nx = nim->nx;
  r_matrix->ny = nim->ny;
  r_matrix->nz = nim->nz;
  r_matrix->matrix = new float[r_matrix->row * r_matrix->col];
  short* data_short = NULL;
  unsigned short* data_ushort = NULL;
  int* data_int = NULL;
  float* data_float = NULL;
  double* data_double = NULL;
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
      FATAL("wrong data type of data file! " << nim->datatype);
  }
#pragma omp parallel for
  // the follow if statements don't harm the performance, it's no significant
  // benifit to put the follow ifs to the previous switch
  for (int i = 0; i < r_matrix->row * r_matrix->col; i++) {
    int t1 = i / r_matrix->row;
    int t2 = i % r_matrix->row;
    if (data_short != NULL) {
      r_matrix->matrix[t2 * r_matrix->col + t1] =
          (float)data_short[i];  // transpose, because data is time (row) *
                                 // voxel (column), r_matrix wants voxel (row) *
                                 // time (column)
    }
    if (data_ushort != NULL) {
      r_matrix->matrix[t2 * r_matrix->col + t1] =
          (float)data_ushort[i];  // transpose, because data is time (row) *
                                  // voxel (column), r_matrix wants voxel (row)
                                  // * time (column)
    }
    if (data_int != NULL) {
      r_matrix->matrix[t2 * r_matrix->col + t1] =
          (float)data_int[i];  // transpose, because data is time (row) * voxel
                               // (column), r_matrix wants voxel (row) * time
                               // (column)
    }
    if (data_float != NULL) {
      r_matrix->matrix[t2 * r_matrix->col + t1] =
          (float)data_float[i];  // transpose, because data is time (row) *
                                 // voxel (column), r_matrix wants voxel (row) *
                                 // time (column)
    }
    if (data_double != NULL) {
      r_matrix->matrix[t2 * r_matrix->col + t1] =
          (float)data_double[i];  // transpose, because data is time (row) *
                                  // voxel (column), r_matrix wants voxel (row)
                                  // * time (column)
    }
  }
  nifti_image_free(nim);
  return r_matrix;
}

/************************************************
Generate masked matrices using the mask file
input: the raw data array, the number of subjects, the mask file, bool value to determine if delete the raw data
output: a masked data array
*************************************************/
RawMatrix** GetMaskedMatrices(RawMatrix** r_matrices, int nSubs,
                              const char* maskFile, bool deleteData) {
  RawMatrix** masked_matrices = new RawMatrix* [nSubs];
  nifti_image* nim = nifti_image_read(maskFile, 1);
  int i;
  if (nim == NULL) {
    FATAL("file not found: ");
  }
  assert(nim);
  int* data_int = NULL;
  short* data_short = NULL;
  unsigned char* data_uchar = NULL;
  float* data_float = NULL;
  double* data_double = NULL;
  // cout<<nim->nx<<" "<<nim->ny<<" "<<nim->nz<<endl; FATAL("debug");
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
      FATAL("wrong data type of mask file!");
  }
#pragma omp parallel for
  for (i = 0; i < nSubs; i++)  // get only the masked voxels
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
    masked_matrix->matrix = new float[row * col];
    float* src_mat = r_matrices[i]->matrix;
    float* dest_mat = masked_matrix->matrix;
    int count = 0;
    for (int j = 0; j < row; j++) {
      if (data_int != NULL && data_int[j]) {
        memcpy(&(dest_mat[count * col]), &(src_mat[j * col]),
               col * sizeof(float));
        count++;
      }
      if (data_short != NULL && data_short[j]) {
        memcpy(&(dest_mat[count * col]), &(src_mat[j * col]),
               col * sizeof(float));
        count++;
      }
      if (data_uchar != NULL && data_uchar[j]) {
        memcpy(&(dest_mat[count * col]), &(src_mat[j * col]),
               col * sizeof(float));
        count++;
      }
      if (data_float != NULL && data_float[j] >= 1 - TINYNUM) {
        memcpy(&(dest_mat[count * col]), &(src_mat[j * col]),
               col * sizeof(float));
        count++;
      }
      if (data_double != NULL && data_double[j] >= 1 - TINYNUM) {
        memcpy(&(dest_mat[count * col]), &(src_mat[j * col]),
               col * sizeof(float));
        count++;
      }
    }
    masked_matrix->row = count;  // update the row information
    masked_matrices[i] = masked_matrix;
    if (deleteData) {
      delete src_mat;
    }
  }
  nifti_image_free(nim);
  return masked_matrices;
}

/************************************************
Generate a masked matrix using the mask file
input: the raw data matrix, the mask file
output: a masked matrix
*************************************************/
RawMatrix* GetMaskedMatrix(RawMatrix* r_matrix, const char* maskFile) {
  RawMatrix* masked_matrix = new RawMatrix;
  nifti_image* nim = nifti_image_read(maskFile, 1);
  int j;
  if (nim == NULL) {
    FATAL("file not found: " << maskFile);
  }
  assert(nim);  // quiet static analyzer
  int* data_int = NULL;
  short* data_short = NULL;
  unsigned char* data_uchar = NULL;
  float* data_float = NULL;
  double* data_double = NULL;
  // cout<<nim->nx<<" "<<nim->ny<<" "<<nim->nz<<endl; FATAL("debug");
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
      FATAL("wrong data type of mask file!");
  }
  masked_matrix->sid = r_matrix->sid;
  masked_matrix->row = r_matrix->row;
  masked_matrix->col = r_matrix->col;
  masked_matrix->nx = r_matrix->nx;
  masked_matrix->ny = r_matrix->ny;
  masked_matrix->nz = r_matrix->nz;
  int row = r_matrix->row;
  int col = r_matrix->col;
  masked_matrix->matrix = new float[row * col];
  float* src_mat = r_matrix->matrix;
  float* dest_mat = masked_matrix->matrix;
  int count = 0;
  for (j = 0; j < row; j++) {
    if (data_int != NULL && data_int[j]) {
      memcpy(&(dest_mat[count * col]), &(src_mat[j * col]),
             col * sizeof(float));
      count++;
    }
    if (data_short != NULL && data_short[j]) {
      memcpy(&(dest_mat[count * col]), &(src_mat[j * col]),
             col * sizeof(float));
      count++;
    }
    if (data_uchar != NULL && data_uchar[j]) {
      memcpy(&(dest_mat[count * col]), &(src_mat[j * col]),
             col * sizeof(float));
      count++;
    }
    if (data_float != NULL && data_float[j] >= 1 - TINYNUM) {
      memcpy(&(dest_mat[count * col]), &(src_mat[j * col]),
             col * sizeof(float));
      count++;
    }
    if (data_double != NULL && data_double[j] >= 1 - TINYNUM) {
      memcpy(&(dest_mat[count * col]), &(src_mat[j * col]),
             col * sizeof(float));
      count++;
    }
    masked_matrix->row = count;  // update the row information
  }
  nifti_image_free(nim);
  return masked_matrix;
}

/************************************************
Generate masked point location using the mask file
input: the raw point location, the number of voxels after masking, the mask file
output: a masked point location array
*************************************************/
VoxelXYZ* GetMaskedPts(VoxelXYZ* pts, int nMaskedVoxels, const char* maskFile) {
  VoxelXYZ* masked_pts = new VoxelXYZ[nMaskedVoxels];
  nifti_image* nim = nifti_image_read(maskFile, 1);
  int i;
  if (nim == NULL) {
    FATAL("file not found: " << maskFile);
  }
  assert(nim);  // quiet static analyzer
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
      FATAL("wrong data type of mask file!");
  }
  int nTotalVoxels = nim->nx * nim->ny * nim->nz;
  int count = 0;
  for (i = 0; i < nTotalVoxels; i++)  // get only the masked voxels
  {
    if (data_int != NULL && data_int[i]) {
      masked_pts[count].x = pts[i].x;
      masked_pts[count].y = pts[i].y;
      masked_pts[count].z = pts[i].z;
      count++;
    }
    if (data_short != NULL && data_short[i]) {
      masked_pts[count].x = pts[i].x;
      masked_pts[count].y = pts[i].y;
      masked_pts[count].z = pts[i].z;
      count++;
    }
    if (data_uchar != NULL && data_uchar[i]) {
      masked_pts[count].x = pts[i].x;
      masked_pts[count].y = pts[i].y;
      masked_pts[count].z = pts[i].z;
      count++;
    }
    if (data_float != NULL && data_float[i] >= 1 - TINYNUM) {
      masked_pts[count].x = pts[i].x;
      masked_pts[count].y = pts[i].y;
      masked_pts[count].z = pts[i].z;
      count++;
    }
    if (data_double != NULL && data_double[i] >= 1 - TINYNUM) {
      masked_pts[count].x = pts[i].x;
      masked_pts[count].y = pts[i].y;
      masked_pts[count].z = pts[i].z;
      count++;
    }
  }
  nifti_image_free(nim);
  return masked_pts;
}

/******************************
Generate trial information for same trials across subjects
input: the number of subjects, the shift number, the number of trials, the block
information file
output: the trial data structure array, the number of trials
*******************************/
Trial* GenRegularTrials(int nSubs, int nShift, int& nTrials, const char* file) {
  std::ifstream ifile(file);
  if (!ifile) {
    FATAL("no block file found!");
  }
  int nPerSubs = -1;
  ifile >> nPerSubs;
  Trial* trials = new Trial[nSubs * nPerSubs];  // 12 trials per subject
  int trial_labels[nPerSubs];  // = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int scs[nPerSubs];  // = {4, 24, 44, 64, 84, 104, 124, 144, 164, 184, 204,
                      // 224};
  int ecs[nPerSubs];  // = {15, 35, 55, 75, 95, 115, 135, 155, 175, 195, 215,
                      // 235};
  // int trial_labels[6] = {0, 1, 0, 1, 0, 1};
  // int scs[6] = {4, 24, 84, 104, 164, 184};
  // int ecs[6] = {15, 35, 95, 115, 175, 195};
  int i, j;
  for (i = 0; i < nPerSubs; i++) {
    ifile >> trial_labels[i] >> scs[i] >> ecs[i];
  }
  ifile.close();
  for (i = 0; i < nSubs; i++) {
    for (j = 0; j < nPerSubs; j++) {
      int index = i * nPerSubs + j;
      trials[index].sid = i;
      trials[index].label = trial_labels[j];
      trials[index].sc = scs[j] + nShift;
      trials[index].ec = ecs[j] + nShift;
      trials[index].tid_withinsubj = j;
      if (i < nSubs / 2) {
        trials[index].tid = j;
      } else {
        if (j % 2 == 0) {
          trials[index].tid = j + 1;  // bds:nPerSubs must be even?
        } else {
          trials[index].tid = j - 1;
        }
      }
    }
  }
  nTrials = nSubs * nPerSubs;
  return trials;
}

/******************************
Generate block information from a block directory, one-to-one mapping to the
data file.
The block files are in txt extension
input: the number of subjects, the shift number, the number of trials, the block
information directory
output: the trial data structure array, the number of trials
*******************************/
Trial* GenBlocksFromDir(int nSubs, int nShift, int& nTrials,
                        RawMatrix** r_matrices, const char* dir) {
  DIR* pDir;
  // cout << dir << endl;
  if ((pDir = opendir(dir)) == NULL) {
    FATAL("invalid block information directory");
  }
  closedir(pDir);
  std::string dirStr = std::string(dir);
  if (dirStr[dirStr.length() - 1] != '/') {
    dirStr += '/';
  }
  int i, j;
  nTrials = 0;
  Trial* trials = new Trial[nSubs * MAXTRIALPERSUBJ];
  int index = 0;
  for (i = 0; i < nSubs; i++) {
    std::string blockFileStr = r_matrices[i]->sname + ".txt";
    blockFileStr = dirStr + blockFileStr;
    std::ifstream ifile(blockFileStr.c_str());
    if (!ifile) {
      FATAL("no block file found!");
    }
    int nPerSubs = -1;
    ifile >> nPerSubs;
    int trial_labels[nPerSubs];
    int scs[nPerSubs];
    int ecs[nPerSubs];
    for (j = 0; j < nPerSubs; j++) {
      ifile >> trial_labels[j] >> scs[j] >> ecs[j];
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
read the location information, from a binary file, the file has row (#voxels)
and column (#TRs) in the beginning, followed by every voxel's 3D location
input: the file name
output: the location struct
****************************/
VoxelXYZ* ReadLocInfo(const char* file) {
  int row, col;
  FILE* fp = fopen(file, "rb");
  fread((void*)&row, sizeof(int), 1, fp);
  fread((void*)&col, sizeof(int), 1, fp);
  VoxelXYZ* pts = new VoxelXYZ[row];
  fread((void*)pts, sizeof(VoxelXYZ), row, fp);
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
generrate the initial location information following the nifti format, using the
nx ny nz information getting from the data nifti files
input: an arbitrary raw matrix, assuming that all subjects have the same format
output: the location struct
****************************/
VoxelXYZ* ReadLocInfoFromNii(RawMatrix* r_matrix) {
  int row = r_matrix->row;
  int nx = r_matrix->nx;
  int ny = r_matrix->ny;
  int nz = r_matrix->nz;
  VoxelXYZ* pts = new VoxelXYZ[row];
  for (int z = 0; z < nz; z++)
    for (int y = 0; y < ny; y++)
      for (int x = 0; x < nx; x++) {
        pts[z * ny * nx + y * nx + x].x = x;
        pts[z * ny * nx + y * nx + x].y = y;
        pts[z * ny * nx + y * nx + x].z = z;
      }
  return pts;
}

/*******************************
read the RT matrices information
input: the file name
output: the RT matrix array, the number of matrices (in the inputting parameter)
********************************/
double** ReadRTMatrices(const char* file, int& nSubs) {
  int nBlocksPerSub = -1;
  FILE* fp = fopen(file, "rb");
  fread((void*)&nSubs, sizeof(int), 1, fp);
  fread((void*)&nBlocksPerSub, sizeof(int), 1, fp);
  double** rtMatrices = new double* [nSubs];
  int i;
  for (i = 0; i < nSubs; i++) {
    double* rtMatrix = new double[nBlocksPerSub * nBlocksPerSub];
    fread((void*)rtMatrix, sizeof(double), nBlocksPerSub * nBlocksPerSub, fp);
    rtMatrices[i] = rtMatrix;
  }
  return rtMatrices;
}

/*******************************
write data to a compressed nifti file
input: the nifti data file name (with or without extension), the sample nifti
file to refer to, the data, the data type(defined in nifti1.h)
output: write the data to the file
********************************/
void WriteNiiGzData(const char* outputFile, const char* refFile, void* data,
                    int dataType) {
  nifti_image* nim =
      nifti_image_read(refFile, 1);  // 1 means reading the data as well
  if (nim == NULL) {
    FATAL("sample file not found: " << refFile);
  }
  assert(nim);  // quiet static analyzer
  nifti_image* nim2 = nifti_copy_nim_info(nim);
  char* newFileName = nifti_makeimgname(
      (char*)outputFile, nim->nifti_type, 0,
      1);  // 3rd argument: 0 means overwrite the existing file, 1 means
           // returning error if the file exists; 4th argument: 0 means not
           // compressed, 1 means compressed
  delete nim2->fname;  // delete the old filename to avoid memory leak
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
input: the nifti data file name (with or without extension), the sample nifti
file to refer to, the data, the data type(defined in nifti1.h), the time (4th)
dimension
output: write the data to the file
********************************/
void Write4DNiiGzData(const char* outputFile, const char* refFile, void* data,
                      int dataType, int nt) {
  nifti_image* nim =
      nifti_image_read(refFile, 1);  // 1 means reading the data as well
  if (nim == NULL) {
    FATAL("sample file not found: " << refFile);
  }
  assert(nim);  // quiet static analyzer
  nifti_image* nim2 = nifti_copy_nim_info(nim);
  char* newFileName = nifti_makeimgname(
      (char*)outputFile, nim->nifti_type, 0,
      1);  // 3rd argument: 0 means overwrite the existing file, 1 means
           // returning error if the file exists; 4th argument: 0 means not
           // compressed, 1 means compressed
  delete nim2->fname;  // delete the old filename to avoid memory leak
  nim2->fname = newFileName;
  nim2->datatype = dataType;
  nim2->nbyper = getSizeByDataType(dataType);
  nim2->nt = nt;
  nim2->dim[0] = 4;
  nim2->dim[4] = nt;
  nim2->nvox = nim2->nx * nim2->ny * nim2->nz * nim2->nt;
  nim2->data = data;
  nifti_image_write(nim2);
  nifti_image_free(nim);
  nifti_image_free(nim2);
  return;
}

/*******************************
generate data to be written to a nifti file based on a mask file, the input data
is specifically from VoxelScore struct
input: the mask nifti file name, the data, the length of the data array, the
data type(defined in nifti1.h, DT_SIGNED_INT indicates to generate voxel id
file, DT_FLOAT indicates to generate voxel score file)
output: the data that is ready to be written to a nifti file
********************************/
void* GenerateNiiDataFromMask(const char* maskFile, VoxelScore* scores,
                              int length, int dataType) {
  nifti_image* nim =
      nifti_image_read(maskFile, 1);  // 1 means reading the data as well
  if (nim == NULL) {
    FATAL("mask file not found: " << maskFile << " in GenerateNiiDataFromMask");
  }
  int* data_int = NULL;
  short* data_short = NULL;
  unsigned char* data_uchar = NULL;
  float* data_float = NULL;
  double* data_double = NULL;
  switch (nim->datatype) {
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
      FATAL("wrong data type of mask file!"
            << " in GenerateNiiDataFromMask");
  }
  int nVoxels = nim->nx * nim->ny * nim->nz;
  void* returnData = NULL;
  int* returnData_int = NULL;
  float* returnData_float = NULL;
  if (dataType == DT_SIGNED_INT) {
    returnData_int = new int[nVoxels];
    memset(returnData_int, 0, nVoxels * sizeof(int));
    returnData = (void*)returnData_int;
  } else if (dataType == DT_FLOAT) {
    returnData_float = new float[nVoxels];
    memset(returnData_float, 0, nVoxels * sizeof(float));
    returnData = (void*)returnData_float;
  } else {
    FATAL("wrong data type of return data!"
          << " in GenerateNiiDataFromMask");
  }
  // generate data from scores
  void* maskedData = new int[length];  // int and float have the same data size
  if (dataType == DT_SIGNED_INT) {
#pragma omp parallel for
    for (int i = 0; i < length; i++) {
      ((int*)maskedData)[scores[i].vid] = i + 1;  // voxel list is 0-based
    }
  } else  // must be DT_FLOAT here
  {
#pragma omp parallel for
    for (int i = 0; i < length; i++) {
      ((float*)maskedData)[scores[i].vid] =
          scores[i].score;  // voxel list is 0-based
    }
  }
  // generate data done

  int count = 0;
  // because of the count variable, no OMP here
  for (int i = 0; i < nVoxels; i++) {
    if (data_int != NULL && data_int[i]) {
      if (count == length) {
        FATAL("number of scores is larger than number of masked voxels "
              << length << "!"
              << " in GenerateNiiDataFromMask");
      }
      if (dataType == DT_SIGNED_INT) {
        returnData_int[i] = ((int*)maskedData)[count];
      } else  // must be DT_FLOAT here
      {
        returnData_float[i] = ((float*)maskedData)[count];
      }
      count++;
    }
    if (data_short != NULL && data_short[i]) {
      if (count == length) {
        FATAL("number of scores is larger than number of masked voxels "
              << length << "!"
              << " in GenerateNiiDataFromMask");
      }
      if (dataType == DT_SIGNED_INT) {
        returnData_int[i] = ((int*)maskedData)[count];
      } else  // must be DT_FLOAT here
      {
        returnData_float[i] = ((float*)maskedData)[count];
      }
      count++;
    }
    if (data_uchar != NULL && data_uchar[i]) {
      if (count == length) {
        FATAL("number of scores is larger than number of masked voxels "
              << length << "!"
              << " in GenerateNiiDataFromMask");
      }
      if (dataType == DT_SIGNED_INT) {
        returnData_int[i] = ((int*)maskedData)[count];
      } else  // must be DT_FLOAT here
      {
        returnData_float[i] = ((float*)maskedData)[count];
      }
      count++;
    }
    if (data_float != NULL && data_float[i] >= 1 - TINYNUM) {
      if (count == length) {
        FATAL("number of scores is larger than number of masked voxels "
              << length << "!"
              << " in GenerateNiiDataFromMask");
      }
      if (dataType == DT_SIGNED_INT) {
        returnData_int[i] = ((int*)maskedData)[count];
      } else  // must be DT_FLOAT here
      {
        returnData_float[i] = ((float*)maskedData)[count];
      }
      count++;
    }
    if (data_double != NULL && data_double[i] >= 1 - TINYNUM) {
      if (count == length) {
        FATAL("number of scores is larger than number of masked voxels "
              << length << "!"
              << " in GenerateNiiDataFromMask");
      }
      if (dataType == DT_SIGNED_INT) {
        returnData_int[i] = ((int*)maskedData)[count];
      } else  // must be DT_FLOAT here
      {
        assert(returnData_float);
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
inline int getSizeByDataType(int datatype) {
  int nbyter = -1;
  switch (datatype) {
    case 2:        // DT_UNSIGNED_CHAR, DT_UINT8
    case 256:      // DT_INT8
      nbyter = 1;  // 1 byte
      break;
    case 4:        // DT_SIGNED_SHORT, DT_INT16
    case 512:      // DT_UINT16
      nbyter = 2;  // 2 bytes
      break;
    case 8:        // DT_SIGNED_INT, DT_INT32
    case 768:      // DT_UINT32
    case 16:       //  DT_FLOAT, DT_FLOAT32
      nbyter = 4;  // 4 bytes
      break;
    case 64:  // DT_DOUBLE, DT_FLOAT64
      nbyter = 8;
      break;
    default:
      FATAL("wrong data type of nifti file in getSizeByDataType!");
  }
  return nbyter;
}

/* get filename extension starting at first '.' */
const char* GetFilenameExtension(const char* filename) {
  const char* dot = strrchr(filename, '.');
  if (!dot || dot == filename) return "";
  return dot + 1;
}

/* read all bytes from a text file into a buffer, which it allocates
    caller should delete buffer when done
 */
char* getTextBytes(const char* fcma_file, long* num_bytes) {
  FILE* f = fopen(fcma_file, "rb");
  fseek(f, 0, SEEK_END);

  long len = ftell(f);
  fseek(f, 0, SEEK_SET);

  *num_bytes = len + 1;
  char* data = new char[*num_bytes];
  fread(data, 1, len, f);
  fclose(f);

  return data;
}

/*******************************
 Read key:value or key=value" pairs from a file, skipping '#' comments.
   - (Caller should allocate "char* keys_and_values[length]"  or "new
 char*[length]")
 input: the filename and array of pointers to hold the values (preceded by max
 array length).
        Keys will be the even indices (0,2,4..) Values the odd indices
 (1,3,5...)
 output: the number of elements (pairs*2-- the keys plus the values)
 ********************************/
int ReadConfigFile(const char* fcma_file, const int& length,
                   char** keys_and_values) {

  const char* lsep = "\n\r";
  const char* sep = ":=";
  char* line, *word, *brkt, *brkw;
  long num_bytes;

  char* text = getTextBytes(fcma_file, &num_bytes);

  int count = 0;
  for (line = strtok_r(text, lsep, &brkt); line;
       line = strtok_r(NULL, lsep, &brkt)) {

    if (line[0] == '#') continue;

    for (word = strtok_r(line, sep, &brkw); word;
         word = strtok_r(NULL, sep, &brkw)) {

      size_t wlen = strlen(word);
      keys_and_values[count] = new char[wlen + 1];
      // printf("%d: got label %s (len %lu)\n",count,word,strlen(word));
      assert(count < length);
      memmove(keys_and_values[count], word, wlen + 1);
      count++;
    }
  }
  delete[] text;
  return count;
}

/* get filename prefix, without extension */
/* Caller must free returned string */
static char* GetFilenamePrefixAndFreeResult(const char* mystr) {
    char *retstr;
    char *lastdot;
    if (mystr == NULL)
        return NULL;
    if ((retstr = (char*)malloc (strlen (mystr) + 1)) == NULL)
        return NULL;
    strcpy (retstr, mystr);
    lastdot = strrchr (retstr, '.');
    if (lastdot != NULL)
        *lastdot = '\0';
    return retstr;
}

/*******************************
 write 4D data to an HDF5 file
 input: the matrix dimensions and float matrix, and outputfile name
 output: none, writes the data to HDF5 file
 ********************************/
void WriteCorrMatToHDF5(int row1, int row2, float* corrMat, const char* outputfile) {
    hid_t       file_id;
    hsize_t     dims[2];
    
    dims[0] = row1;
    dims[1] = row2;
    
    char* prefix = GetFilenamePrefixAndFreeResult(outputfile);
    char h5out[MAXFILENAMELENGTH];
    sprintf(h5out,"%s.h5", prefix);
    free(prefix);
    
    /* create a HDF5 file */
    file_id = H5Fcreate (h5out, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    /* create and write an integer type dataset named dset */
    H5LTmake_dataset(file_id, "/dset", 2, dims, H5T_NATIVE_FLOAT, corrMat);
    
    /* close file */
    H5Fclose (file_id);
}

#endif

