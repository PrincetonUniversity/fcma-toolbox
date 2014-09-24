#include "FisherScoring.h"
#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif 

// data only cotains one row, i.e. one voxel's data
void GetSAndH(float* data, float* data2, double* beta, int length, int* labels, int rank, double* S, double* H, int v1, int v2, int n)
{
  int i;
  memset((void*)S, 0, rank*sizeof(double));
  memset((void*)H, 0, rank*rank*sizeof(double));
  if (rank==2)
  {
    data2=NULL;  // pretend to use this variable
    for (i=0; i<length; i++)
    {
      int y = labels[i];
      double X[] = {1.0, data[i]};
      double exponent = X[0]*beta[0]+X[1]*beta[1];
      double e = exp(exponent);
      double coef = e/(1+e);
      double coef2 = e/((1+e)*(1+e));
      S[0] += y*X[0]-coef*X[0];
      S[1] += y*X[1]-coef*X[1];
      H[0] -= coef2*X[0]*X[0];
      H[1] -= coef2*X[0]*X[1];
      H[2] = H[1];
      H[3] -= coef2*X[1]*X[1];
    }
  }
  else if (rank==3)
  {
    __declspec(align(64)) double s0[length];
    __declspec(align(64)) double s1[length];
    __declspec(align(64)) double s2[length];
    __declspec(align(64)) double h0[length];
    __declspec(align(64)) double h1[length];
    __declspec(align(64)) double h2[length];
    __declspec(align(64)) double h4[length];
    __declspec(align(64)) double h5[length];
    __declspec(align(64)) double h8[length];
    #pragma simd
    for (i=0; i<length; i++)
    {
      int y = labels[i];
      //double X[] = {1.0, data[i], data2[i]};
      double exponent = beta[0]+data[i]*beta[1]+data2[i]*beta[2];
      double e = exp(exponent);
      double coef = e/(1+e);
      double coef2 = coef/(1+e);
      //if (isinf(e)) {coef=1.0; coef2=0.0;}  // this branch harms the performance of MIC more than host
      s0[i] = y-coef;
      s1[i] = y*data[i]-coef*data[i];
      s2[i] = y*data2[i]-coef*data2[i];
      h0[i] = coef2;//coef2*X[0]*X[0];
      h1[i] = coef2*data[i];//coef2*X[0]*X[1];
      h2[i] = coef2*data2[i];//coef2*X[0]*X[2];
      h4[i] = h1[i]*data[i];//coef2*X[1]*X[1];
      h5[i] = h1[i]*data2[i];//coef2*X[1]*X[2];
      h8[i] = h2[i]*data2[i];//coef2*X[2]*X[2];
    }
    // in the vectorization+reduction case, if the compiler is not sure about the dependency (do S and H overlap?),
    // it cannot be vectorized even with #pragma ivdep
    #pragma ivdep
    for (i=0; i<length; i++)
    {
      S[0] += s0[i];
      S[1] += s1[i];
      S[2] += s2[i];
    }
    #pragma ivdep
    for (i=0; i<length; i++)
    {
      H[0] -= h0[i];
      H[1] -= h1[i];
      H[2] -= h2[i];
      H[4] -= h4[i];
      H[5] -= h5[i];
      H[8] -= h8[i];
    }
    H[3] = H[1];
    H[6] = H[2];
    H[7] = H[5];
  }
}

double* GetInverseMat(double* mat, int rank)
{
  double* iMat = new double[rank*rank];
  if (rank==2)
  {
    double det = mat[0*rank+0]*mat[1*rank+1]-mat[0*rank+1]*mat[1*rank+0];
    if (det==0)
    {
      cerr<<mat[0*rank+0]<<" "<<mat[0*rank+1]<<endl;
      cerr<<mat[1*rank+0]<<" "<<mat[1*rank+1]<<endl;
      cerr<<"matrix cannot be inversed!"<<endl;
      exit(1);
    }
    iMat[0*rank+0] = mat[1*rank+1]/det;
    iMat[0*rank+1] = -mat[0*rank+1]/det;
    iMat[1*rank+0] = -mat[1*rank+0]/det;
    iMat[1*rank+1] = mat[0*rank+0]/det;
  }
  else if (rank==3)
  {
    double a=mat[0];
    double b=mat[1];
    double c=mat[2];
    double d=mat[4];
    double e=mat[5];
    double f=mat[8];
    double det = -1/(a*d*f-a*e*e-b*b*f+2*b*c*e-c*c*d);
    if (det==0)
    {
      cerr<<a<<" "<<b<<" "<<c<<" "<<d<<endl;
      cerr<<mat[0]<<" "<<mat[1]<<" "<<mat[2]<<" "<<mat[4]<<" "<<mat[5]<<" "<<mat[8]<<endl;
      cerr<<"matrix cannot be inversed!"<<endl;
      det=1;
    }
    iMat[0*rank+0]=(e*e-d*f)*det;
    iMat[0*rank+1]=(b*f-c*e)*det;
    iMat[0*rank+2]=(c*d-b*e)*det;
    iMat[1*rank+0]=iMat[0*rank+1];
    iMat[1*rank+1]=(c*c-a*f)*det;
    iMat[1*rank+2]=(a*e-b*c)*det;
    iMat[2*rank+0]=iMat[0*rank+2];
    iMat[2*rank+1]=iMat[1*rank+2];
    iMat[2*rank+2]=(b*b-a*d)*det;
  }
  return iMat;
}

double DoIteration(float* data, float* data2, int length, int* labels, double epsilon, int rank, int v1, int v2)
{
  //double* beta=new double[rank];
  double* beta = (double*)_mm_malloc(sizeof(double)*rank, 64);
  memset((void*)beta, 0, rank*sizeof(double));
  //double* S=new double[rank];
  double* S = (double*)_mm_malloc(sizeof(double)*rank, 64);
  //double* H=new double[rank*rank];
  double* H = (double*)_mm_malloc(sizeof(double)*rank*rank, 64);
  double loglikelihood = 0.0;
  if (rank==2)
  {
    double lambda = epsilon+1;
    double* iH;
    while (lambda>epsilon)
    {
      GetSAndH(data, data2, beta, length, labels, rank, S, H, v1, v2, -1);
      if (H[2]==0 && H[3]==0)
      { for (int ii=0; ii<length; ii++) cerr<<data[ii]<<" "; cerr<<endl;cerr<<epsilon<<endl;}
      iH = GetInverseMat(H, rank);
      lambda = -((S[0]*iH[0]+S[1]*iH[2])*S[0]+(S[0]*iH[1]+S[1]*iH[3])*S[1]);
      beta[0] = beta[0]-(iH[0]*S[0]+iH[1]*S[1]);
      beta[1] = beta[1]-(iH[2]*S[0]+iH[3]*S[1]);
      delete iH;
    }
    for (int i=0; i<length; i++)
    {
      int y = labels[i];
      double X[] = {1.0, data[i]};
      double exponent = X[0]*beta[0]+X[1]*beta[1];
      double e = exp(exponent);
      loglikelihood += y*exponent-log(1+e);
    }
  }
  else if (rank==3)
  {
    double lambda = epsilon+1;
    double* iH;
    int n=0;
    while (lambda>epsilon)
    {
      GetSAndH(data, data2, beta, length, labels, rank, S, H, v1, v2, n);
      iH = GetInverseMat(H, rank);
      lambda = -((S[0]*iH[0]+S[1]*iH[3]+S[2]*iH[6])*S[0]
               + (S[0]*iH[1]+S[1]*iH[4]+S[2]*iH[7])*S[1]
               + (S[0]*iH[2]+S[1]*iH[5]+S[2]*iH[8])*S[2]);
      beta[0] = beta[0]-(iH[0]*S[0]+iH[1]*S[1]+iH[2]*S[2]);
      beta[1] = beta[1]-(iH[3]*S[0]+iH[4]*S[1]+iH[5]*S[2]);
      beta[2] = beta[2]-(iH[6]*S[0]+iH[7]*S[1]+iH[8]*S[2]);
      delete iH;
      n++;
    }
    //#pragma simd
    for (int i=0; i<length; i++)
    {
      int y = labels[i];
      double X[] = {1.0, data[i], data2[i]};
      double exponent = X[0]*beta[0]+X[1]*beta[1]+X[2]*beta[2];
      double e = exp(exponent);
      loglikelihood += y*exponent-log(1+e);
    }
  }
  _mm_free(beta);
  _mm_free(H);
  _mm_free(S);
  //delete beta;
  //delete H;
  //delete S;
  return loglikelihood;
}

double DoIteration2(float* dataHead, int offset1, float* dataHead2, int offset2, int length, int* labels, double epsilon, int rank)
{
  float* data = dataHead+offset1;
  float* data2 = dataHead2+offset2;
  double* beta=new double[rank];
  memset((void*)beta, 0, rank*sizeof(double));
  double* S=new double[rank];
  double* H=new double[rank*rank];
  double loglikelihood = 0.0;
  if (rank==2)
  {
    double lambda = epsilon+1;
    double* iH;
    while (lambda>epsilon)
    {
      GetSAndH(data, data2, beta, length, labels, rank, S, H, -1, -1, -1);
      if (H[2]==0 && H[3]==0)
      { for (int ii=0; ii<length; ii++) cerr<<data[ii]<<" "; cerr<<endl;cerr<<epsilon<<endl;}
      iH = GetInverseMat(H, rank);
      lambda = -((S[0]*iH[0]+S[1]*iH[2])*S[0]+(S[0]*iH[1]+S[1]*iH[3])*S[1]);
      beta[0] = beta[0]-(iH[0]*S[0]+iH[1]*S[1]);
      beta[1] = beta[1]-(iH[2]*S[0]+iH[3]*S[1]);
      delete iH;
    }
    for (int i=0; i<length; i++)
    {
      int y = labels[i];
      double X[] = {1.0, data[i]};
      double exponent = X[0]*beta[0]+X[1]*beta[1];
      double e = exp(exponent);
      loglikelihood += y*exponent-log(1+e);
    }
  }
  else if (rank==3)
  {
    double lambda = epsilon+1;
    double* iH;
    while (lambda>epsilon)
    {
      GetSAndH(data, data2, beta, length, labels, rank, S, H, -1, -1, -1);
      iH = GetInverseMat(H, rank);
      lambda = -((S[0]*iH[0]+S[1]*iH[3]+S[2]*iH[6])*S[0]
               + (S[0]*iH[1]+S[1]*iH[4]+S[2]*iH[7])*S[1]
               + (S[0]*iH[2]+S[1]*iH[5]+S[2]*iH[8])*S[2]);
      beta[0] = beta[0]-(iH[0]*S[0]+iH[1]*S[1]+iH[2]*S[2]);
      beta[1] = beta[1]-(iH[3]*S[0]+iH[4]*S[1]+iH[5]*S[2]);
      beta[2] = beta[2]-(iH[6]*S[0]+iH[7]*S[1]+iH[8]*S[2]);
      delete iH;
    }
    #pragma simd
    for (int i=0; i<length; i++)
    {
      int y = labels[i];
      double X[] = {1.0, data[i], data2[i]};
      double exponent = X[0]*beta[0]+X[1]*beta[1]+X[2]*beta[2];
      double e = exp(exponent);
      loglikelihood += y*exponent-log(1+e);
    }
  }
  delete beta;
  delete H;
  delete S;
  return loglikelihood;
}
