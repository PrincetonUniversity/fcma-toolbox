#include "Classification.h"
#include "Scheduler.h"
#include "CorrMatAnalysis.h"
#include "common.h"

/****************************************
get the distance ratios between correlation vectors within the same category and across categories for every voxel
the distance ratio to be computed here is not based on the Euclidean distance but something that is similar to the square of Euclidean distance
input: the node id, the correlation matrix array, the number of blocks
output: a list of voxels' scores in terms of average distance ratios
*****************************************/
VoxelScore* GetDistanceRatio(int me, CorrMatrix** c_matrices, int nTrials)  //get scores for a c_matrix array
{
  if (me==0)  //sanity check
  {
    cerr<<"the master node isn't supposed to do classification jobs"<<endl;
    exit(1);
  }
  int rowBase = c_matrices[0]->sr;  // assume all elements in c_matrices array have the same starting row
  int row = c_matrices[0]->nVoxels; // assume all elements in c_matrices array have the same #voxels
  int length = row * c_matrices[0]->step; // assume all elements in c_matrices array have the same step, get the number of entries of a coorelation matrix, notice the row here!!
  VoxelScore* scores = new VoxelScore[c_matrices[0]->step];  // get step voxels classification accuracy here
  #pragma omp parallel for
  for (int i=0; i<length; i+=row)
  {
    int count = i / row;  // write count in this way for parallelization
    (scores+count)->vid = rowBase+i/row;
    (scores+count)->score = 1 - DoDistanceRatioSmarter(nTrials, i, c_matrices, row);
  }
  return scores;
}

/*****************************************
get the distance ratio between correlation vectors within the same category and across categories for one voxel
input: the number of blocks, the voxel id, the correlation matrix array, the number of entries in a partial correlation matrix
output: the distance ratio of this voxel
******************************************/
float DoDistanceRatioSmarter(int nTrainings, int startIndex, CorrMatrix** c_matrices, int length)
{
  int i, j, nIn0=0, nIn1=0;
  float* c0 = new float[length];
  float* c0_2 = new float[length];
  float* c1 = new float[length];
  float* c1_2 = new float[length];
  for (i=0; i<length; i++)
  {
    c0[i] = 0.0;
    c0_2[i] = 0.0;
    c1[i] = 0.0;
    c1_2[i] = 0.0;
  }
  for (i=0; i<nTrainings; i++)
  {
    if (c_matrices[i]->tlabel == 1)
    {
      for (j=0; j<length; j++)
      {
        c0[j] += c_matrices[i]->matrix[startIndex+j];
        c0_2[j] += c_matrices[i]->matrix[startIndex+j] * c_matrices[i]->matrix[startIndex+j];
      }
      nIn0++;
    }
    else
    {
      for (j=0; j<length; j++)
      {
        c1[j] += c_matrices[i]->matrix[startIndex+j];
        c1_2[j] += c_matrices[i]->matrix[startIndex+j] * c_matrices[i]->matrix[startIndex+j];
      }
      nIn1++;
    }
  }
  float* sum_c0 = new float[length];
  float* sum_c1 = new float[length];
  float* sum_c01 = new float[length];
  for (i=0; i<length; i++)
  {
    sum_c0[i] = nIn0*c0_2[i] - c0[i]*c0[i];
    if (sum_c0[i]<0) sum_c0[i]=0.0;//{cout<<"sum_c0: "<<nIn0<<" "<<c0[i]<<" "<<c0_2[i]<<" "<<sum_c0[i]; getchar();}
    sum_c1[i] = nIn1*c1_2[i] - c1[i]*c1[i];
    if (sum_c1[i]<0) sum_c1[i]=0.0;//{cout<<"sum_c1: "<<nIn1<<" "<<c1[i]<<" "<<c1_2[i]<<" "<<sum_c1[i]; getchar();}
    sum_c01[i] = nIn1*c0_2[i] + nIn0*c1_2[i] - 2*c0[i]*c1[i];
    if (sum_c01[i]<0) sum_c01[i]=0.0;//{cout<<"sum_c01: "<<nIn0<<" "<<nIn1<<" "<<" "<<c0[i]<<" "<<c0_2[i]<<" "<<c1[i]<<" "<<c1_2[i]<<" "<<sum_c01[i]; getchar();}
  }
  float dIn0 = GetVectorSum(sum_c0, length);
  float dIn1 = GetVectorSum(sum_c1, length);
  float dOut = GetVectorSum(sum_c01, length);
  //cout<<((dIn0+dIn1)/((nIn0-1)*nIn0/2+(nIn1-1)*nIn1/2))<<" "<<(dOut/(nIn0*nIn1)); getchar();
  float ratio = ((dIn0+dIn1)/((nIn0-1)*nIn0/2+(nIn1-1)*nIn1/2)) / (dOut/(nIn0*nIn1));
  delete c0;  // try to use array instead of new later
  delete c0_2;
  delete c1;
  delete c1_2;
  delete sum_c0;
  delete sum_c1;
  delete sum_c01;
  return ratio;
}

/*float GetVectorSum(float* v, int length)
{
  int i;
  float result = 0.0;
  for (i=0; i<length; i++)
  {
    result += v[i]; // could add sqrt here sqrt(v[i])
  }
  return result;  // or here sqrt(result)
}*/
