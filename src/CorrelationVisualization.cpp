#include "CorrelationVisualization.h"
#include "FileProcessing.h"
#include "MatComputation.h"
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
void VisualizeCorrelationWithMasks(RawMatrix* r_matrix, const char* maskFile1, const char* maskFile2, Trial trial, const char* output_file)
{
  RawMatrix* masked_matrix1=NULL;
  RawMatrix* masked_matrix2=NULL;
  if (maskFile1!=NULL)
    masked_matrix1 = GetMaskedMatrix(r_matrix, maskFile1);
  else
    masked_matrix1 = r_matrix;
  if (maskFile2!=NULL)
    masked_matrix2 = GetMaskedMatrix(r_matrix, maskFile2);
  else
    masked_matrix2 = r_matrix;
  cout<<"masked matrix generating done!"<<endl;
  cout<<"#voxels for mask1: "<<masked_matrix1->row<<" #voxels for mask2: "<<masked_matrix2->row<<endl;
  double* mat1 = masked_matrix1->matrix;
  double* mat2 = masked_matrix2->matrix;
  int row1 = masked_matrix1->row;
  int row2 = masked_matrix2->row;
  int col = masked_matrix1->col;
  float* buf1 = new float[row1*col]; // col is more than what really need, just in case
  float* buf2 = new float[row2*col]; // col is more than what really need, just in case
  int sc=trial.sc, ec=trial.ec;
  int ml1 = getBuf(sc, ec, row1, col, mat1, buf1);  // get the normalized matrix, return the length of time points to be computed
  int ml2 = getBuf(sc, ec, row2, col, mat2, buf2);  // get the normalized matrix, return the length of time points to be computed, m1==m2
  float* corrMat = new float[row1*row2];
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, row1, row2, ml1, 1.0, buf1, ml1, buf2, ml2, 0.0, corrMat, row2);
  delete buf1;
  delete buf2;
  float* wholeData = PutMaskedDataBack(maskFile2, corrMat, row1, row2);
  Write4DNiiGzData(output_file, "/state/partition3/yidawang/face_scene/0309101_conatt_localizer_std_bet.nii.gz", (void*)wholeData, DT_FLOAT32, row1);
  return;
}

/*******************************
generate data to be written to a nifti file based on a mask file, the input data is specifically from VoxelScore struct
input: the mask nifti file name, the data (assuming to be float), the row of the data array (corresponding to the 4th dimension), the column of the data array (corresponding to the mask)
output: the data that is ready to be written to a nifti file
********************************/
float* PutMaskedDataBack(const char* maskFile, float* data, int row, int col)
{
  nifti_image* nim = nifti_image_read(maskFile, 1); // 1 means reading the data as well
  if (nim == NULL)
  {
    cerr<<"mask file not found: "<<maskFile<<" in PutMaskedDataBack"<<endl;
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
      cerr<<"wrong data type of mask file!"<<" in PutMaskedDataBack"<<endl;
      exit(1);
  }
  int nVoxels = nim->nx*nim->ny*nim->nz;
  float* returnData = new float[nVoxels*row];
  memset(returnData, 0, nVoxels*row*sizeof(float));

  int count=0;
  // because of the count variable, no OMP here
  for (int i=0; i<row; i++)
  {
    for (int j=0; j<nVoxels; j++)
    {
      if (data_int!=NULL && data_int[j])
      {
        if (count==col*row)
        {
          cerr<<"number of scores is larger than number of masked voxels "<<col*row<<"!"<<" in PutMaskedDataBack"<<endl;
          exit(1);
        }
        returnData[i*nVoxels+j] = data[count];
        count++;
      }
      if (data_short!=NULL && data_short[j])
      {
        if (count==col*row)
        {
          cerr<<"number of scores is larger than number of masked voxels "<<col*row<<"!"<<" in PutMaskedDataBack"<<endl;
          exit(1);
        }
        returnData[i*nVoxels+j] = data[count];
        count++;
      }
      if (data_uchar!=NULL && data_uchar[j])
      {
        if (count==col*row)
        {
          cerr<<"number of scores is larger than number of masked voxels "<<col*row<<"!"<<" in PutMaskedDataBack"<<endl;
          exit(1);
        }
        returnData[i*nVoxels+j] = data[count];
        count++;
      }
      if (data_float!=NULL && data_float[j]>=1-TINYNUM)
      {
        if (count==col*row)
        {
          cerr<<"number of scores is larger than number of masked voxels "<<col*row<<"!"<<" in PutMaskedDataBack"<<endl;
          exit(1);
        }
        returnData[i*nVoxels+j] = data[count];
        count++;
      }
      if (data_double!=NULL && data_double[j]>=1-TINYNUM)
      {
        if (count==col*row)
        {
          cerr<<"number of scores is larger than number of masked voxels "<<col*row<<"!"<<" in PutMaskedDataBack"<<endl;
          exit(1);
        }
        returnData[i*nVoxels+j] = data[count];
        count++;
      }
    }
  }
  nifti_image_free(nim);
  return returnData;
}
