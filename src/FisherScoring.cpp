#include "FisherScoring.h"
#include "common.h"

// data only cotains one row, i.e. one voxel's data
void GetSAndH(float* data, float* data2, float* beta, int length, int* labels, int rank, float* S, float* H, int v1, int v2, int n)
{
  int i;
  memset((void*)S, 0, rank*sizeof(float));
  memset((void*)H, 0, rank*rank*sizeof(float));
  if (rank==2)
  {
    data2=NULL;  // pretend to use this variable
    for (i=0; i<length; i++)
    {
      int y = labels[i];
      float X[] = {1.0, data[i]};
      float exponent = X[0]*beta[0]+X[1]*beta[1];
      float e = exp(exponent);
      float coef = e/(1+e);
      float coef2 = e/((1+e)*(1+e));
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
    //if (n>1000) {cout<<n<<endl; exit(0);}
    ALIGNED(64) float s0[length];
    ALIGNED(64) float s1[length];
    ALIGNED(64) float s2[length];
    ALIGNED(64) float h0[length];
    ALIGNED(64) float h1[length];
    ALIGNED(64) float h2[length];
    ALIGNED(64) float h4[length];
    ALIGNED(64) float h5[length];
    ALIGNED(64) float h8[length];
    #pragma simd
    for (i=0; i<length; i++)
    {
      int y = labels[i];
      //float X[] = {1.0, data[i], data2[i]};
      float exponent = beta[0]+data[i]*beta[1]+data2[i]*beta[2];
      float e = exp(exponent);
      float coef = e/(1+e);
      float coef2 = coef/(1+e);
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

float* GetInverseMat(float* mat, int rank)
{
  int lwork = 102;
  float work[lwork];
  int info;
  float wr[rank], wi[rank];
  float tempMat[rank*rank];
  memcpy((void*)tempMat, (const void*)mat, rank*rank*sizeof(float));
  //for (int i=0; i<rank*rank; i++) tempMat[i] = -tempMat[i];
  //dgeev( "N", "N", &rank, tempMat, &rank, wr, wi, NULL, &rank, NULL, &rank, work, &lwork, &info );
  // ridge regularization
  /*float delta=-0.1;
  for (int i=0; i<rank; i++)
  {
    if (wr[i]>delta)
    {
      mat[0] += delta;
      mat[4] += delta;
      if (rank==3) mat[8] += delta;
      break;
    }
  }*/
  /*if (rank==3){
  cout<<mat[0]<<" "<<mat[1]<<" "<<mat[2]<<endl;
  cout<<mat[3]<<" "<<mat[4]<<" "<<mat[5]<<endl;
  cout<<mat[6]<<" "<<mat[7]<<" "<<mat[8]<<endl;
  cout<<wr[0]<<" "<<wr[1]<<" "<<wr[2]<<endl;exit(1);}*/
  float* iMat = new float[rank*rank];
  if (rank==2)
  {
    float det = mat[0*rank+0]*mat[1*rank+1]-mat[0*rank+1]*mat[1*rank+0];
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
    float a=mat[0];
    float b=mat[1];
    float c=mat[2];
    float d=mat[4];
    float e=mat[5];
    float f=mat[8];
    float det = -1/(a*d*f-a*e*e-b*b*f+2*b*c*e-c*c*d);
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

float DoIteration(float* data, float* data2, int length, int* labels, float epsilon, int rank, int v1, int v2)
{
  //float* beta=new float[rank];
  float* beta = (float*)_mm_malloc(sizeof(float)*rank, 64);
  memset((void*)beta, 0, rank*sizeof(float));
  //float* S=new float[rank];
  float* S = (float*)_mm_malloc(sizeof(float)*rank, 64);
  //float* H=new float[rank*rank];
  float* H = (float*)_mm_malloc(sizeof(float)*rank*rank, 64);
  float loglikelihood = 0.0;
  if (rank==2)
  {
    float lambda = epsilon+1;
    float* iH;
    while (lambda>epsilon)
    {
      GetSAndH(data, data2, beta, length, labels, rank, S, H, v1, v2, -1);
      iH = GetInverseMat(H, rank);
      lambda = -((S[0]*iH[0]+S[1]*iH[2])*S[0]+(S[0]*iH[1]+S[1]*iH[3])*S[1]);
      beta[0] = beta[0]-(iH[0]*S[0]+iH[1]*S[1]);
      beta[1] = beta[1]-(iH[2]*S[0]+iH[3]*S[1]);
      delete [] iH; // bds []
    }
    for (int i=0; i<length; i++)
    {
      int y = labels[i];
      float X[] = {1.0, data[i]};
      float exponent = X[0]*beta[0]+X[1]*beta[1];
      float e = exp(exponent);
      loglikelihood += y*exponent-log(1+e);
    }
  }
  else if (rank==3)
  {
    float lambda = epsilon+1;
    float* iH;
    int n=0;
    while (lambda>epsilon)
    {
      GetSAndH(data, data2, beta, length, labels, rank, S, H, v1, v2, n);
      //cout<<data[0]<<" "<<data[1]<<" "<<data[2]<<endl;
      //cout<<data2[0]<<" "<<data2[1]<<" "<<data2[2]<<endl;
      iH = GetInverseMat(H, rank);
      lambda = -((S[0]*iH[0]+S[1]*iH[3]+S[2]*iH[6])*S[0]
               + (S[0]*iH[1]+S[1]*iH[4]+S[2]*iH[7])*S[1]
               + (S[0]*iH[2]+S[1]*iH[5]+S[2]*iH[8])*S[2]);
      beta[0] = beta[0]-(iH[0]*S[0]+iH[1]*S[1]+iH[2]*S[2]);
      beta[1] = beta[1]-(iH[3]*S[0]+iH[4]*S[1]+iH[5]*S[2]);
      beta[2] = beta[2]-(iH[6]*S[0]+iH[7]*S[1]+iH[8]*S[2]);
      delete [] iH; // bds
      n++;
    }
    #pragma simd reduction(+:loglikelihood)
    for (int i=0; i<length; i++)
    {
      int y = labels[i];
      float X[] = {1.0, data[i], data2[i]};
      float exponent = X[0]*beta[0]+X[1]*beta[1]+X[2]*beta[2];
      float e = exp(exponent);
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

float DoIteration2(float* dataHead, int offset1, float* dataHead2, int offset2, int length, int* labels, float epsilon, int rank)
{
  float* data = dataHead+offset1;
  float* data2 = dataHead2+offset2;
  float* beta=new float[rank];
  memset((void*)beta, 0, rank*sizeof(float));
  float* S=new float[rank];
  float* H=new float[rank*rank];
  float loglikelihood = 0.0;
  if (rank==2)
  {
    float lambda = epsilon+1;
    float* iH;
    while (lambda>epsilon)
    {
      GetSAndH(data, data2, beta, length, labels, rank, S, H, -1, -1, -1);
      if (H[2]==0 && H[3]==0)
      { for (int ii=0; ii<length; ii++) cerr<<data[ii]<<" "; cerr<<endl;cerr<<epsilon<<endl;}
      iH = GetInverseMat(H, rank);
      lambda = -((S[0]*iH[0]+S[1]*iH[2])*S[0]+(S[0]*iH[1]+S[1]*iH[3])*S[1]);
      beta[0] = beta[0]-(iH[0]*S[0]+iH[1]*S[1]);
      beta[1] = beta[1]-(iH[2]*S[0]+iH[3]*S[1]);
      delete [] iH; // bds []
    }
    for (int i=0; i<length; i++)
    {
      int y = labels[i];
      float X[] = {1.0, data[i]};
      float exponent = X[0]*beta[0]+X[1]*beta[1];
      float e = exp(exponent);
      loglikelihood += y*exponent-log(1+e);
    }
  }
  else if (rank==3)
  {
    float lambda = epsilon+1;
    float* iH;
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
      delete [] iH; // bds []
    }
    #pragma simd reduction(+:loglikelihood)
    for (int i=0; i<length; i++)
    {
      int y = labels[i];
      float X[] = {1.0, data[i], data2[i]};
      float exponent = X[0]*beta[0]+X[1]*beta[1]+X[2]*beta[2];
      float e = exp(exponent);
      loglikelihood += y*exponent-log(1+e);
    }
  }
  delete [] beta;   // bds []
  delete [] H;      // "
  delete [] S;      // "
  return loglikelihood;
}
