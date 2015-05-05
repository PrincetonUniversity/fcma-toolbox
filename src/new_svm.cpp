#include "new_svm.h"
#include "ErrorHandling.h"

#ifndef Calloc
#define Calloc(type,n) (type *)calloc(n,sizeof(type))
#endif
// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
// works only for binary classification now, i.e. two classes
void svmGroupClasses(int nPoints, float *labels, int **start_ret, int **count_ret, int *perm)
{
  int l = nPoints;
  int nr_class = 2;
  int *count = Calloc(int,nr_class);
  int data_label[l];	
  int i;
  for(i=0;i<l;i++)
  {
    if ((int)labels[i]==1)
    {
      count[0]++;
      data_label[i] = 0;
    }
    else
    {
      count[1]++;
      data_label[i] = 1;
    }
  }
  int *start = Calloc(int,nr_class);
  start[0] = 0;
  start[1] = count[0];
  for(i=0;i<l;i++)
  {
    perm[start[data_label[i]]] = i;
    ++start[data_label[i]];
  }
  start[0] = 0;
  start[1] = count[0];

  *start_ret = start;
  *count_ret = count;
}

//float* data, int nPoints, int nDimension, float* labels, float** p_alpha, Kernel_params* kp, float cost, SelectionHeuristic heuristicMethod, float epsilon, float tolerance, float* transposedData
// only works for binary classification now, i.e. two classes
float crossValidationNoShuffle(float* data, int nPoints, int nDimension, int nFolds, float* labels, Kernel_params* kp, float cost, SelectionHeuristic heuristicMethod, float epsilon, float tolerance, float* transposedData, int vid)
{
/**
  get kernel type
**/
  int kType = NEWGAUSSIAN;
  float parameterA;
  if (kp->kernel_type.compare(0,3,"rbf") == 0) {
    parameterA = -kp->gamma;
    kType = NEWGAUSSIAN;
    //printf("Gaussian kernel: gamma = %f\n", -parameterA);
  } else if (kp->kernel_type.compare(0,6,"linear") == 0) {
    kType = NEWLINEAR;
    //printf("Linear kernel\n");
  } else if (kp->kernel_type.compare(0,11,"precomputed") == 0) {
    kType = NEWPRECOMPUTED;
    //printf("Precomputed kernel\n");
  }
#ifdef __SVM_BREAKDOWN__
  struct timeval start_time, end_time;
  float t0=0, t1=0, t2=0, t3=0;
  gettimeofday(&start_time, 0);
#endif
/**
  group folds
**/
  //float* target=(float*)kmp_malloc(sizeof(float)*nPoints);
  float target[nPoints];
  int i;
  int fold_start[nFolds+1];
  int perm[nPoints];
  int nr_class=2;
  // stratified cv may not give leave-one-out rate
  // Each class to l folds -> some folds may have zero elements
  if(nFolds < nPoints)
  {
    int *start = NULL;
    int *count = NULL;
    // put the same class to be adjacent to each other
    svmGroupClasses(nPoints, labels, &start, &count, perm);

    // data grouped by fold using the array perm
    int *fold_count = Calloc(int,nFolds);
    int c;
    int *index = Calloc(int,nPoints);
    for(i=0;i<nPoints;i++)
      index[i]=perm[i];
    // according to the number of folds, assign each fold with same number of different classes
    for(i=0;i<nFolds;i++)
    {
      fold_count[i] = 0;
      for (c=0; c<nr_class;c++)
        fold_count[i]+=(i+1)*count[c]/nFolds-i*count[c]/nFolds;
    }
    fold_start[0]=0;
    for (i=1;i<=nFolds;i++)
      fold_start[i] = fold_start[i-1]+fold_count[i-1];
    for (c=0; c<nr_class;c++)
      for(i=0;i<nFolds;i++)
      {
        int begin = start[c]+i*count[c]/nFolds;
        int end = start[c]+(i+1)*count[c]/nFolds;
        for(int j=begin;j<end;j++)
        {
          perm[fold_start[i]] = index[j];
          fold_start[i]++;
        }
      }
    fold_start[0]=0;
    for (i=1;i<=nFolds;i++)
      fold_start[i] = fold_start[i-1]+fold_count[i-1];
    free(start);	
    free(count);	
    free(index);
    free(fold_count);
  }
  else
  {
    FATAL("too many folds in SVM cross validation");
  }

/**
  cross validation
**/
  float* sub_data = (float*)malloc(sizeof(float)*nPoints*nDimension);  // trade off space to avoid allocating spaces multiple times
  float* sub_labels = (float*)malloc(sizeof(float)*nPoints);  // ditto
  float* test_data = (float*)malloc(sizeof(float)*nPoints*nDimension);
  float* test_labels = (float*)malloc(sizeof(float)*nPoints);
  bool *bit_map;
  float sv_alpha[nPoints];  // normally nSV<<sub_nPoints
  float supportVectors[nPoints*nDimension];
  if (kType==NEWPRECOMPUTED)
  {
    bit_map = Calloc(bool,nPoints);
  }
#ifdef __SVM_BREAKDOWN__
  gettimeofday(&end_time, 0);
  t0 = end_time.tv_sec-start_time.tv_sec+(end_time.tv_usec-start_time.tv_usec)*0.000001;
#endif
  for(i=0;i<nFolds;i++)
  {
#ifdef __SVM_BREAKDOWN__
    gettimeofday(&start_time, 0);
#endif
    int begin = fold_start[i];
    int end = fold_start[i+1];    
    int j,k,l;
    int sub_nPoints = nPoints - (end-begin);
    if (kType==NEWPRECOMPUTED)
    {
      for (j=0; j<begin; j++)
      {
        bit_map[perm[j]]=true;
      }
      for (j=begin; j<end; j++)
      {
        bit_map[perm[j]]=false;
      }
      for (j=end; j<nPoints; j++)
      {
        bit_map[perm[j]]=true;
      }
    }
    k=0;
    for(j=0;j<begin;j++)
    {
      if (kType==NEWPRECOMPUTED) // take the left-out data out, inefficient now
      {
        int k1=0;
        for (l=0; l<nPoints; l++)
        {
          if (bit_map[perm[l]])
          {
            sub_data[k*sub_nPoints+k1]=data[perm[j]*nDimension+perm[l]];
            k1++;
          }
        }
      }
      else      // deep copy the data
      {
        memcpy((void*)(sub_data+k*nDimension), (const void*)(data+perm[j]*nDimension), sizeof(float)*nDimension);
      }
      sub_labels[k] = labels[perm[j]]==0?-1.0f:1.0f;
      ++k;
    }
    for(j=end;j<nPoints;j++)
    {
      if (kType==NEWPRECOMPUTED) // take the left-out data out, inefficient now
      {
        int k1=0;
        for (l=0; l<nPoints; l++)
        {
          if (bit_map[perm[l]])
          {
            sub_data[k*sub_nPoints+k1]=data[perm[j]*nDimension+perm[l]];
            k1++;
          }
        }
      }
      else      // deep copy the data
      {
        memcpy((void*)(sub_data+k*nDimension), (const void*)(data+perm[j]*nDimension), sizeof(float)*nDimension);
      }
      sub_labels[k] = labels[perm[j]]==0?-1.0f:1.0f;
      ++k;
    }
#ifdef __SVM_BREAKDOWN__
    gettimeofday(&end_time, 0);
    t1+=end_time.tv_sec-start_time.tv_sec+(end_time.tv_usec-start_time.tv_usec)*0.000001;
    gettimeofday(&start_time, 0);
#endif
    float* alpha;
    int sub_nDimension = nDimension;
    if (kType==NEWPRECOMPUTED)
    {
      sub_nDimension = sub_nPoints;
    }
    performTraining(sub_data, sub_nPoints, sub_nDimension, sub_labels, &alpha, kp, cost, heuristicMethod, epsilon, tolerance, transposedData, i, vid);
/**
   generate support vectors
**/
    int nSV=0;
    //float* sv_alpha=(float*)kmp_malloc(sizeof(float)*sub_nPoints);  // normally nSV<<sub_nPoints
    //float* supportVectors=(float*)kmp_malloc(sizeof(float)*sub_nPoints*sub_nDimension);
    int SV_index=0;
    for (j=0; j<sub_nPoints;j++)
    {
      if (alpha[j] > epsilon)
      {
        sv_alpha[nSV]=alpha[j]*sub_labels[j];
        if (kType==NEWPRECOMPUTED)
        {
          supportVectors[SV_index++]=j;
        }
        else
        {
          memcpy((void*)(supportVectors+nSV*sub_nDimension), (const void*)(sub_data+j*sub_nDimension), sizeof(float)*sub_nDimension);
        }
        nSV++;
      }
		}
#ifdef __SVM_BREAKDOWN__
    gettimeofday(&end_time, 0);
    t2+=end_time.tv_sec-start_time.tv_sec+(end_time.tv_usec-start_time.tv_usec)*0.000001;
#endif

/**
  generate test_data and test_labels
**/
#ifdef __SVM_BREAKDOWN__
    gettimeofday(&start_time, 0);
#endif
    int nTests = end-begin;
    k=0;
    for (j=begin; j<end; j++)
    {
      if (kType==NEWPRECOMPUTED) // take the left-out data out, inefficient now
      {
        int k1=0;
        for (l=0; l<nPoints; l++)
        {
          if (bit_map[perm[l]])
          {
            test_data[k*sub_nPoints+k1]=data[perm[j]*nDimension+perm[l]];
            k1++;
          }
        }
      }
      else      // deep copy the data
      {
        memcpy((void*)(test_data+k*nDimension), (const void*)(data+perm[j]*nDimension), sizeof(float)*nDimension);
      }
      test_labels[k] = labels[perm[j]]==0?-1.0f:1.0f;
      ++k;
    }
    float* result;
    performClassification(test_data, nTests, supportVectors, nSV, sub_nDimension, sv_alpha, kp, &result);
    //#pragma simd
    for(j=begin;j<end;j++)
    {
      target[perm[j]] = result[j-begin];
    }
    free(alpha);
    free(result);
    //kmp_free(sv_alpha);
    //kmp_free(supportVectors);
#ifdef __SVM_BREAKDOWN__
    gettimeofday(&end_time, 0);
    t3+=end_time.tv_sec-start_time.tv_sec+(end_time.tv_usec-start_time.tv_usec)*0.000001;
#endif
  }
#ifdef __SVM_BREAKDOWN__
  if (omp_get_thread_num()==0)
  {
    printf("SVM time breakdown:\n");
    printf("preparing folds: %fs\n", t0);
    printf("generating training/test sets: %fs\n", t1);
    printf("training: %fs\n", t2);
    printf("test: %fs\n", t3);
  }
#endif
  int nCorrects=0;
  for (i=0; i<nPoints; i++)
  {
    nCorrects=(labels[i]==1&&target[i]>=0)||(labels[i]==0&&target[i]<0)?nCorrects+1:nCorrects;
  }
  free(sub_data);
  free(sub_labels);
  free(test_data);
  free(test_labels);
  if (kType==NEWPRECOMPUTED)
  {
    free(bit_map);
  }
  return 1.0*nCorrects/nPoints;
}

void performTraining(float* data, int nPoints, int nDimension, float* labels, float** p_alpha, Kernel_params* kp, float cost, SelectionHeuristic heuristicMethod, float epsilon, float tolerance, float* transposedData, int fold_id, int vid) {
  
  float cEpsilon = cost - epsilon;
  Controller progress(2.0, heuristicMethod, 64, nPoints);

  int kType = NEWGAUSSIAN;
  float parameterA;
  float parameterB;
  float parameterC;
  if (kp->kernel_type.compare(0,3,"rbf") == 0) {
    parameterA = -kp->gamma;
    kType = NEWGAUSSIAN;
    printf("Gaussian kernel: gamma = %f\n", -parameterA);
  } else if (kp->kernel_type.compare(0,6,"linear") == 0) {
    kType = NEWLINEAR;
    printf("Linear kernel\n");
  } else if (kp->kernel_type.compare(0,11,"precomputed") == 0) {
    kType = NEWPRECOMPUTED;
    //printf("Precomputed kernel\n");
  }
  //printf("Cost: %f, Tolerance: %f, Epsilon: %f\n", cost, tolerance, epsilon);
  
  float* devData;
  float* devTransposedData;
  size_t devDataPitch;
  size_t devTransposedDataPitch;
  int hostPitchInFloats = nPoints;
  float* hostData;
  bool hostDataAlloced = false;

  devData = data;
  int devDataPitchInFloats = nPoints;
  devTransposedData = transposedData;
  int devTransposedDataPitchInFloats = nDimension;
  float* devLabels = labels;
  float devKernelDiag[nPoints];
  float* alpha = (float*)malloc(sizeof(float) * nPoints);
  *p_alpha = alpha;
  float* devAlpha = alpha;
  float devF[nPoints];
  float* hostResult = (float*)malloc(8*sizeof(float));
  void* devResult = (void*) hostResult;
  int blockWidth = intDivideRoundUp(nPoints, BLOCKSIZE);

  size_t rowPitch = nPoints*sizeof(float);  
  size_t sizeOfCache = (10*1024*1024*1024L)/(rowPitch); //10GB cache
  if (nPoints < sizeOfCache) {
    sizeOfCache = nPoints;
  }
		
  //printf("%Zu rows of kernel matrix will be cached (%Zu bytes per row)\n", sizeOfCache, rowPitch);

  char* str_num_threads = getenv("OMP_NUM_THREADS");
  int nthreads = (str_num_threads == NULL)?(1):(atoi(str_num_threads));
  if (nthreads <= 0) nthreads = 1;
  //printf("Number of threads = %d \n", nthreads);

  float* devCache;
  size_t cachePitch = rowPitch;
  NewCache kernelCache(nPoints, sizeOfCache);
  int devCachePitchInFloats = (int)cachePitch/(sizeof(float));

  //emulate precomputed kernel..
  if (kType == NEWPRECOMPUTED) {
    if(nPoints != nDimension) { 
	printf("Not a kernel matrix (not square) \n"); exit(1); 
    }
    devCache = devData;
    //memcpy(devCache, devData, nPoints*nPoints*sizeof(float));
    //for (int i = 0 ;i < nPoints*nPoints; i++) devCache[i] /= 1000.0f;
    //printf("Cache populated\n");
  }
  else {
    devCache = new float[sizeOfCache * nPoints];
  }
  launchInitialization(devData, devDataPitchInFloats, devCache, devCachePitchInFloats, nPoints, nDimension, kType, parameterA, parameterB, parameterC, devKernelDiag, devAlpha, devF, devLabels, nthreads);
  //printf("Initialization complete\n");

  //Choose initial points
  float bLow = 1;
  float bHigh = -1;
  int iteration = 0;
  int iLow = -1;
  int iHigh = -1;
  for (int i = 0; i < nPoints; i++) {
    if (labels[i] < 0) {
      if (iLow == -1) {
        iLow = i;
        if (iHigh > -1) {
          i = nPoints; //Terminate
        }
      }
    } else {
      if (iHigh == -1) {
        iHigh = i;
        if (iLow > -1) {
          i = nPoints; //Terminate
        }
      }
    }
  }
  launchTakeFirstStep(devResult, devKernelDiag, devData, devDataPitchInFloats, devCache, devCachePitchInFloats, devAlpha, cost, nDimension, iLow, iHigh, kType, parameterA, parameterB, parameterC, nthreads);

  float alpha2Old = *(hostResult + 0);
  float alpha1Old = *(hostResult + 1);
  bLow = *(hostResult + 2);
  bHigh = *(hostResult + 3);
  float alpha2New = *(hostResult + 6);
  float alpha1New = *(hostResult + 7);

  float alpha1Diff = alpha1New - alpha1Old;
  float alpha2Diff = alpha2New - alpha2Old;
	
  int iLowCacheIndex;
  int iHighCacheIndex;
  bool iLowCompute;
  bool iHighCompute; 
  
  //printf("Starting iterations\n");
  for (iteration = 1; true; iteration++) {
    if (bLow <= bHigh + 2*tolerance || (bLow==1.0f && bHigh==-1.0f && iteration>=200)) {
      //printf("Converged\n");
      break; //Convergence!!
    }

    if ((iteration & 0x7ff) == 0) {
      //printf("iteration: %d; gap: %f\n",iteration, bLow - bHigh);
    }
        
    if ((iteration & 0x7f) == 0) {
      heuristicMethod = progress.getMethod();
    }

    if (kType != NEWPRECOMPUTED) { 
      kernelCache.findData(iHigh, iHighCacheIndex, iHighCompute);
      kernelCache.findData(iLow, iLowCacheIndex, iLowCompute);
    }

    if (heuristicMethod == FIRSTORDER) {
      launchFirstOrder(iLowCompute, iHighCompute, kType, nPoints, nDimension, devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, epsilon, cEpsilon, devAlpha, devF, alpha1Diff * labels[iHigh], alpha2Diff * labels[iLow], iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devKernelDiag, devResult, cost, nthreads);
    } else {
      launchSecondOrder(iLowCompute, iHighCompute, kType, nPoints, nDimension, devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, epsilon, cEpsilon, devAlpha, devF, alpha1Diff * labels[iHigh], alpha2Diff * labels[iLow], iLow, iHigh, parameterA, parameterB, parameterC, &kernelCache, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devKernelDiag, devResult, cost, iteration, nthreads);
    }

    alpha2Old = *(hostResult + 0);
    alpha1Old = *(hostResult + 1);
    bLow = *(hostResult + 2);
    bHigh = *(hostResult + 3);
    iLow = *((int*)hostResult + 6);
    iHigh = *((int*)hostResult + 7);
    alpha2New = *(hostResult + 4);
    alpha1New = *(hostResult + 5);
    alpha1Diff = alpha1New - alpha1Old;
    alpha2Diff = alpha2New - alpha2Old;
    progress.addIteration(bLow-bHigh);    
  }
	
  //printf("%d iterations\n", iteration);
  //printf("bLow: %f, bHigh: %f\n", bLow, bHigh);
  kp->b = (bLow + bHigh) / 2;
  //kernelCache.printStatistics();
  free(hostResult);
  if (kType != NEWPRECOMPUTED) {
    free(devCache);
  }
}

void QP(float* devKernelDiag, float kernelEval, float* devAlpha, float* devLabels, int iHigh, int iLow, float bHigh, float bLow, float cost, void* devResult) {
  
    float eta = devKernelDiag[iHigh] + devKernelDiag[iLow] - 2*kernelEval;
      
    float alpha1Old = devAlpha[iHigh];
    float alpha2Old = devAlpha[iLow];
    float alphaDiff = alpha2Old - alpha1Old;
    float lowLabel = devLabels[iLow];
    float sign = devLabels[iHigh] * lowLabel;
    float alpha2UpperBound;
    float alpha2LowerBound;
    if (sign < 0) {
      if (alphaDiff < 0) {
        alpha2LowerBound = 0;
	alpha2UpperBound = cost + alphaDiff;
      } else {
        alpha2LowerBound = alphaDiff;
        alpha2UpperBound = cost;
      }
    } else {
      float alphaSum = alpha2Old + alpha1Old;
      if (alphaSum < cost) {
        alpha2UpperBound = alphaSum;
        alpha2LowerBound = 0;
      } else {
        alpha2LowerBound = alphaSum - cost;
        alpha2UpperBound = cost;
			}
    }
    float alpha2New;
    if (eta > 0) {
      alpha2New = alpha2Old + lowLabel*(bHigh - bLow)/eta;
      if (alpha2New < alpha2LowerBound) {
        alpha2New = alpha2LowerBound;
      } else if (alpha2New > alpha2UpperBound) {
	alpha2New = alpha2UpperBound;
      }
    } else {
      float slope = lowLabel * (bHigh - bLow);
      float delta = slope * (alpha2UpperBound - alpha2LowerBound);
      if (delta > 0) {
        if (slope > 0) {
          alpha2New = alpha2UpperBound;
        } else {
          alpha2New = alpha2LowerBound;
        }
      } else {
        alpha2New = alpha2Old;
      }
    }
    float alpha2Diff = alpha2New - alpha2Old;
    float alpha1Diff = -sign*alpha2Diff;
    float alpha1New = alpha1Old + alpha1Diff;

    *((float*)devResult + 0) = alpha2Old;
    *((float*)devResult + 1) = alpha1Old;
    *((float*)devResult + 2) = bLow;
    *((float*)devResult + 3) = bHigh;
    devAlpha[iLow] = alpha2New;
    devAlpha[iHigh] = alpha1New;
    *((float*)devResult + 4) = alpha2New;
    *((float*)devResult + 5) = alpha1New;
    *((int*)devResult + 6) = iLow;
    *((int*)devResult + 7) = iHigh;
}

template<int Kernel>
void	firstOrder(float* devData, int devDataPitchInFloats, float* devTransposedData, int devTransposedDataPitchInFloats, float* devLabels, int nPoints, int nDimension, float epsilon, float cEpsilon, float* devAlpha, float* devF, float alpha1Diff, float alpha2Diff, int iLow, int iHigh, float parameterA, float parameterB, float parameterC, float* devCache, int devCachePitchInFloats, int iLowCacheIndex, int iHighCacheIndex, float* devKernelDiag, void* devResult, float cost, bool iLowCompute, bool iHighCompute, int nthreads) {

    float* xILow = devTransposedData + iLow*devTransposedDataPitchInFloats;
    float* xIHigh = devTransposedData + iHigh*devTransposedDataPitchInFloats;

    float* highKernel;
    float* lowKernel;

    if (!iHighCompute) {
	    highKernel = devCache + (devCachePitchInFloats * iHighCacheIndex) ;
    }
    if (!iLowCompute) {
    	lowKernel = devCache + (devCachePitchInFloats * iLowCacheIndex);
    }

    if (iHighCompute) {
      highKernel = new float[nPoints];
      //#pragma omp parallel for num_threads(nthreads)
      for (int j = 0; j < nPoints; j++) {
        float* x = devTransposedData +j*devTransposedDataPitchInFloats;
        float acc = 0.0f;
        for (int i = 0; i < nDimension; i++) {
          float diff = x[i] - xIHigh[i];
          acc += diff*diff;
        }
        highKernel[j] = exp(parameterA * acc);
      }
    }
    if (iLowCompute) {
       lowKernel = new float[nPoints];
       //#pragma omp parallel for num_threads(nthreads)
       for (int j = 0; j < nPoints; j++) {
        float* x = devTransposedData +j*devTransposedDataPitchInFloats;
        float acc = 0.0f;
        for (int i = 0; i < nDimension; i++) {
          float diff =  x[i] - xILow[i];
          acc += diff*diff;
        }
        lowKernel[j] = exp(parameterA * acc);
      }
   }

    //const int CL = 16; //64 bytes
    //reduction arrays
    //int* devLocalIndicesRL = new int[nthreads*CL];
    //int* devLocalIndicesRH = new int[nthreads*CL];
    //float* devLocalFsRL = new float[nthreads*CL]; // init to -inf
    //float* devLocalFsRH = new float[nthreads*CL];  //init to inf
    int devLocalIndicesRL;
    int devLocalIndicesRH;
    float devLocalFsRL = -FLT_MAX; // init to -inf
    float devLocalFsRH = FLT_MAX;  //init to inf
    
    /*for (int tid = 0 ; tid < nthreads; tid++) {
      devLocalFsRL[tid*CL] = -FLT_MAX;
      devLocalFsRH[tid*CL] = FLT_MAX;
    }*/

    //#pragma omp parallel for num_threads(nthreads)
    //bool flag0[nPoints];
    //bool flag1[nPoints];
    float localFsRL[nPoints];
    float localFsRH[nPoints];
    #pragma simd
    for (int globalIndex = 0; globalIndex < nPoints; globalIndex++) {
      int tid = 0;//omp_get_thread_num();

      float alpha = devAlpha[globalIndex];
      float f = devF[globalIndex];
      float label = devLabels[globalIndex];
      f += alpha1Diff * highKernel[globalIndex];
      f += alpha2Diff * lowKernel[globalIndex];
      devCache[(devCachePitchInFloats * iLowCacheIndex) + globalIndex]=iLowCompute?lowKernel[globalIndex]:devCache[(devCachePitchInFloats * iLowCacheIndex) + globalIndex];
      devCache[(devCachePitchInFloats * iHighCacheIndex) + globalIndex]=iHighCompute?highKernel[globalIndex]:devCache[(devCachePitchInFloats * iHighCacheIndex) + globalIndex];

      bool flag0 = (alpha>epsilon && (alpha<cEpsilon || label>0)) || (alpha<=epsilon && label<=0);
      bool flag1 = (alpha>epsilon && (alpha<cEpsilon || label<=0)) || (alpha<=epsilon && label>0);
      devF[globalIndex] = f;
      localFsRL[globalIndex] = flag0?f:-FLT_MAX;
      localFsRH[globalIndex] = flag1?f:FLT_MAX;
    }
    // can be further optimized using intrinsics --Yida
    for (int globalIndex = 0; globalIndex < nPoints; globalIndex++) {
      if (localFsRL[globalIndex] > devLocalFsRL) { devLocalFsRL = localFsRL[globalIndex]; devLocalIndicesRL = globalIndex;}
      if (localFsRH[globalIndex] < devLocalFsRH) { devLocalFsRH = localFsRH[globalIndex]; devLocalIndicesRH = globalIndex;}
    }

    float maxFsRL = devLocalFsRL;
    float minFsRH = devLocalFsRH;
    int iLowNew=devLocalIndicesRL, iHighNew=devLocalIndicesRH;

    /*for (int tid = 0; tid < nthreads; tid++) {
      if (maxFsRL < devLocalFsRL[tid*CL]) { maxFsRL = devLocalFsRL[tid*CL]; iLowNew = devLocalIndicesRL[tid*CL];}
      if (minFsRH > devLocalFsRH[tid*CL]) { minFsRH = devLocalFsRH[tid*CL]; iHighNew = devLocalIndicesRH[tid*CL];}
    }*/
    float bLow = maxFsRL;
    float bHigh = minFsRH;

    if(iHighCompute) delete [] highKernel;
    if(iLowCompute) delete [] lowKernel;
    //delete [] devLocalIndicesRL;
    //delete [] devLocalIndicesRH;
    //delete [] devLocalFsRL;
    //delete [] devLocalFsRH;

    *((float*)devResult + 2) = bLow;
    *((float*)devResult + 3) = bHigh;
    *((int*)devResult + 6) = iLowNew;
    *((int*)devResult + 7) = iHighNew;
}

void launchFirstOrder(bool iLowCompute, bool iHighCompute, int kType, int nPoints, int nDimension, float* devData, int devDataPitchInFloats, float* devTransposedData, int devTransposedDataPitchInFloats, float* devLabels, float epsilon, float cEpsilon, float* devAlpha, float* devF, float sAlpha1Diff, float sAlpha2Diff, int iLow, int iHigh, float parameterA, float parameterB, float parameterC, float* devCache, int devCachePitchInFloats, int iLowCacheIndex, int iHighCacheIndex, float* devKernelDiag, void* devResult, float cost, int nthreads) {

  float kernelEval, bLow, bHigh;
  int iHighNew, iLowNew;

  switch(kType) {
    case NEWGAUSSIAN:
      firstOrder<NEWGAUSSIAN>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devKernelDiag, devResult, cost, iLowCompute, iHighCompute, nthreads);
      bLow = *((float*)devResult + 2);
      bHigh = *((float*)devResult + 3);
      iLowNew = *((int*)devResult + 6);
      iHighNew = *((int*)devResult + 7);
      float* highPointer = devTransposedData + (iHighNew * devTransposedDataPitchInFloats);
      float* lowPointer = devTransposedData + (iLowNew * devTransposedDataPitchInFloats);  
      float acc = 0.0f;
      for (int i = 0; i < nDimension; i++) {
        float diff = highPointer[i] - lowPointer[i];
        acc += diff*diff;
      }
      kernelEval = exp(parameterA * acc);
      break;
    case NEWPRECOMPUTED:
      iHighCompute = false; iHighCacheIndex = iHigh;
      iLowCompute = false; iLowCacheIndex = iLow;
      firstOrder<NEWPRECOMPUTED>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devKernelDiag, devResult, cost, iLowCompute, iHighCompute, nthreads);
      bLow = *((float*)devResult + 2);
      bHigh = *((float*)devResult + 3);
      iLowNew = *((int*)devResult + 6);
      iHighNew = *((int*)devResult + 7);
      kernelEval = devCache[iHighNew * devCachePitchInFloats + iLowNew];
      break;
    default:
      printf("Unsupported kernel \n");
      exit(1);
  }

  //Found iHigh and iLow. Now do QP
  QP(devKernelDiag, kernelEval, devAlpha, devLabels, iHighNew, iLowNew, bHigh, bLow, cost, devResult);
}

template<int Kernel>
void	secondOrder(float* devData, int devDataPitchInFloats, float* devTransposedData, int devTransposedDataPitchInFloats, float* devLabels, int nPoints, int nDimension, float epsilon, float cEpsilon, float* devAlpha, float* devF, float alpha1Diff, float alpha2Diff, int iLow, int iHigh, float parameterA, float parameterB, float parameterC, float* devCache, int devCachePitchInFloats, int iLowCacheIndex, int iHighCacheIndex, float* devKernelDiag, void* devResult, float cost, float bHigh, bool iHighCompute, int nthreads) {

  float* xIHigh = devTransposedData + iHigh*devTransposedDataPitchInFloats;
  float* highKernel;

  if (iHighCompute) {
    highKernel = new float[nPoints];
    //#pragma omp parallel for num_threads(nthreads)
    for (int j = 0; j < nPoints; j++) {
      float* x = devTransposedData +j*devTransposedDataPitchInFloats;
      float acc = 0.0f;
      for (int i = 0; i < nDimension; i++) {
        float diff = x[i] - xIHigh[i];
        acc += diff*diff;
      }
      highKernel[j] = exp(parameterA * acc);
    }
  } else {
    highKernel = devCache + devCachePitchInFloats * iHighCacheIndex;
  }

  float iHighSelfKernel = devKernelDiag[iHigh];

  //const int CL = 16; //64 bytes
  //reduction arrays
  /*int* devLocalIndicesRL = new int[nthreads*CL];
  float* devLocalFsRL = new float[nthreads*CL]; // init to -inf
  for (int tid = 0 ; tid < nthreads; tid++) {
    devLocalFsRL[tid*CL] = -FLT_MAX;
  } */
  int devLocalIndicesRL;
  float devLocalFsRL = -FLT_MAX;

  //#pragma omp parallel for num_threads(nthreads)
  float objs[nPoints];
  #pragma simd
  for(int globalIndex = 0; globalIndex < nPoints; globalIndex++) {
    int tid = 0;//omp_get_thread_num();
    float alpha = devAlpha[globalIndex];
    float label = devLabels[globalIndex];
    float f = devF[globalIndex];
    float beta = bHigh - f;
    devCache[(devCachePitchInFloats * iHighCacheIndex) + globalIndex]=iHighCompute?highKernel[globalIndex]:devCache[(devCachePitchInFloats * iHighCacheIndex) + globalIndex];
    bool flag=((label>0&&alpha>epsilon)||(label<0&&alpha<cEpsilon))&&beta<=epsilon;
    float kappa=iHighSelfKernel+devKernelDiag[globalIndex]-2*highKernel[globalIndex];
    kappa = kappa<=0?epsilon:kappa;
    objs[globalIndex] = flag?beta * beta / kappa:-FLT_MAX;
  }
  for(int globalIndex = 0; globalIndex < nPoints; globalIndex++) {
    if (objs[globalIndex]>devLocalFsRL) {
      devLocalFsRL=objs[globalIndex];
      devLocalIndicesRL = globalIndex;
    }
  }  

  //reduction.. 
  //
    /*float maxFsRL = -FLT_MAX;
    int iLowNew;

    for (int tid = 0; tid < nthreads; tid++) {
        if (maxFsRL < devLocalFsRL[tid*CL]) { maxFsRL = devLocalFsRL[tid*CL]; iLowNew = devLocalIndicesRL[tid*CL];}
    }*/
    float bLow = devF[devLocalIndicesRL];

    if(iHighCompute) delete [] highKernel;

    *((float*)devResult + 2) = bLow;
    *((int*)devResult + 6) = devLocalIndicesRL;

}

void launchSecondOrder(bool iLowCompute, bool iHighCompute, int kType, int nPoints, int nDimension, float* devData, int devDataPitchInFloats, float* devTransposedData, int devTransposedDataPitchInFloats, float* devLabels, float epsilon, float cEpsilon, float* devAlpha, float* devF, float sAlpha1Diff, float sAlpha2Diff, int iLow, int iHigh, float parameterA, float parameterB, float parameterC, NewCache* kernelCache, float* devCache, int devCachePitchInFloats, int iLowCacheIndex, int iHighCacheIndex, float* devKernelDiag, void* devResult, float cost, int iteration, int nthreads) {

  float kernelEval, bLow, bHigh;
  int iHighNew, iLowNew;

  switch(kType) {
    case NEWGAUSSIAN:
      firstOrder<NEWGAUSSIAN>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devKernelDiag, devResult, cost, iLowCompute, iHighCompute, nthreads);

      bHigh = *((float*)devResult + 3);
      iHighNew = *((int*)devResult + 7);

      kernelCache->findData(iHighNew, iHighCacheIndex, iHighCompute);

      secondOrder<NEWGAUSSIAN>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHighNew, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devKernelDiag, devResult, cost, bHigh, iHighCompute, nthreads);

      bLow = *((float*)devResult + 2);
      iLowNew = *((int*)devResult + 6);

      kernelEval = devCache[iHighCacheIndex*devCachePitchInFloats + iLowNew];

      break;

    case NEWPRECOMPUTED:
      iHighCompute = false; iHighCacheIndex = iHigh;
      iLowCompute = false; iLowCacheIndex = iLow;
      firstOrder<NEWPRECOMPUTED>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devKernelDiag, devResult, cost, iLowCompute, iHighCompute, nthreads);

      bHigh = *((float*)devResult + 3);
      iHighNew = *((int*)devResult + 7);

      iHighCacheIndex = iHighNew;
      secondOrder<NEWPRECOMPUTED>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHighNew, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devKernelDiag, devResult, cost, bHigh, iHighCompute, nthreads);

      bLow = *((float*)devResult + 2);
      iLowNew = *((int*)devResult + 6);

      kernelEval = devCache[iHighCacheIndex*devCachePitchInFloats + iLowNew];

      break;

    default:
      printf("Unsupported kernel \n");
      exit(1);
  }
  //Found iHigh and iLow. Now do QP
  QP(devKernelDiag, kernelEval, devAlpha, devLabels, iHighNew, iLowNew, bHigh, bLow, cost, devResult);
  return;
}

NewCache::DirectoryEntry::DirectoryEntry() {
  status = NEVER;
}

NewCache::NewCache(int nPointsIn, int cacheSizeIn) {
  directory.reserve(nPointsIn);
  nPoints = nPointsIn;
  cacheSize = cacheSizeIn;
  occupancy = 0;
  hits = 0;
  compulsoryMisses = 0;
  capacityMisses = 0;
}

NewCache::~NewCache() {
 
}

void NewCache::search(const int index, int &offset, bool &compute) {
  DirectoryEntry currentEntry = directory[index];
  if (currentEntry.status == DirectoryEntry::INCACHE) {
    offset = currentEntry.location;
    compute = false;
    return;
  }
  compute = true;
  return;
}

void NewCache::findData(const int index, int &offset, bool &compute) {
  std::vector<DirectoryEntry>::iterator iCurrentEntry = directory.begin() + index;
  if (iCurrentEntry->status == DirectoryEntry::INCACHE) {
    hits++;
    if (iCurrentEntry->lruListEntry == lruList.begin()) {
      offset = iCurrentEntry->location;
      compute = false;
      return;
    }
    lruList.erase(iCurrentEntry->lruListEntry);
    lruList.push_front(index);
    iCurrentEntry->lruListEntry = lruList.begin();
    offset = iCurrentEntry->location;
    compute = false;
    return;
  }

  //Cache Miss
  if (occupancy < cacheSize) {
    //Cache has empty space
    compulsoryMisses++;
    iCurrentEntry->location = occupancy;
    iCurrentEntry->status = DirectoryEntry::INCACHE;
    lruList.push_front(index);
    iCurrentEntry->lruListEntry = lruList.begin();
    occupancy++;
    offset = iCurrentEntry->location;
    compute = true;
    return;
  }
 
  //Cache is full
  if (iCurrentEntry->status == DirectoryEntry::NEVER) {
    compulsoryMisses++;
  } else {
    capacityMisses++;
  }

  int expiredPoint = lruList.back();
  lruList.pop_back();
 
  directory[expiredPoint].status = DirectoryEntry::EVICTED;
  int expiredLine = directory[expiredPoint].location;
  iCurrentEntry->status = DirectoryEntry::INCACHE;
  iCurrentEntry->location = expiredLine;
  lruList.push_front(index);
  iCurrentEntry->lruListEntry = lruList.begin();

  offset = iCurrentEntry->location;
  compute = true;
  return;
}

void NewCache::printStatistics() {
	int accesses = hits + compulsoryMisses + capacityMisses;
	printf("%d accesses, %d hits, %d compulsory misses, %d capacity misses\n", accesses, hits, compulsoryMisses, capacityMisses);
	return;
}

void NewCache::printCache() {
  int accesses = hits + compulsoryMisses + capacityMisses;
  float hitPercent = (float)hits*100.0/float(accesses);
  float compulsoryPercent = (float)compulsoryMisses*100.0/float(accesses);
  float capacityPercent = (float)capacityMisses*100.0/float(accesses);
  
  printf("Cache hits: %f compulsory misses: %f capacity misses %f\n", hitPercent, compulsoryPercent, capacityPercent);
  for(int i = 0; i < nPoints; i++) {
    if (directory[i].status == DirectoryEntry::INCACHE) {
      printf("Row %d: present @ cache line %d\n", i, directory[i].location);
    } else {
      printf("Row %d: not present\n", i);
    }
  }
  printf("----\n");
  std::list<int>::iterator i = lruList.begin();
  for(;i != lruList.end(); i++) {
    printf("Offset: %d\n", *i);
  }
}

Controller::Controller(float initialGap, SelectionHeuristic currentMethodIn, int samplingIntervalIn, int problemSize) {
  progress.push_back(initialGap);
  currentMethod = currentMethodIn;
  if (currentMethod == ADAPTIVE) {
    adaptive = true;
    currentMethod = SECONDORDER;
  } else {
    adaptive = false;
  }
  samplingInterval = samplingIntervalIn;
  inspectionPeriod = problemSize/(10*samplingInterval);

  
  timeSinceInspection = inspectionPeriod - 2;
  beginningOfEpoch = 0;
  rates.push_back(0);
  rates.push_back(0);
  currentInspectionPhase = 0;
  //printf("Controller: currentMethod: %i (%s), inspectionPeriod: %i\n", currentMethod, adaptive?"dynamic":"static", inspectionPeriod);
}

void Controller::addIteration(float gap) {
  progress.push_back(gap);
  method.push_back(currentMethod);
  
}

float Controller::findRate(struct timeval* start, struct timeval* finish, int beginning, int end) {
  //printf("findRate: (%i -> %i) = ", beginning, end);
  float time = ((float)(finish->tv_sec - start->tv_sec))*1000000 + ((float)(finish->tv_usec - start->tv_usec));
  int length = end - beginning;
  int filterLength = length/2;
  float phase1Gap = filter(beginning, beginning + filterLength);
  float phase2Gap = filter(beginning + filterLength, end);
  float percentageChange = (phase2Gap - phase1Gap)/phase1Gap;
  float percentRate = percentageChange / time;
  //printf("%f\n", percentRate);
  return percentRate;
}

SelectionHeuristic Controller::getMethod() {
  if (!adaptive) {
    if (currentMethod == RANDOM) {
      if ((rand() & 0x1) > 0) {
        return SECONDORDER;
      } else {
        return FIRSTORDER;
      }
    }
    return currentMethod;
  }

  
  if (timeSinceInspection >= inspectionPeriod) {
    int currentIteration = progress.size();
    gettimeofday(&start, 0);
    currentInspectionPhase = 1;
    timeSinceInspection = 0;
    beginningOfEpoch = currentIteration;
  } else if (currentInspectionPhase == 1) {
    int currentIteration = progress.size();

    middleOfEpoch = currentIteration;
    gettimeofday(&mid, 0);
    rates[currentMethod] = findRate(&start, &mid, beginningOfEpoch, middleOfEpoch);
    currentInspectionPhase++;

    if (currentMethod == FIRSTORDER) {
      currentMethod = SECONDORDER;
    } else {
      currentMethod = FIRSTORDER;
    }
    
  } else if (currentInspectionPhase == 2) {
    int currentIteration = progress.size();
        
    gettimeofday(&finish, 0);
    rates[currentMethod] = findRate(&mid, &finish, middleOfEpoch, currentIteration);
    timeSinceInspection = 0;
    currentInspectionPhase = 0;
    
    if (fabs(rates[1]) > fabs(rates[0])) {
      currentMethod = SECONDORDER;
    } else {
      currentMethod = FIRSTORDER;
    }
    //printf("Rate 0: %f, Rate 1: %f, choose method: %i\n", rates[0], rates[1], currentMethod);
  } else {
    timeSinceInspection++;
  }
  return currentMethod;
}

float Controller::filter(int begin, int end) {
  float accumulant = 0;
  for (int i = begin; i < end; i++) {
    accumulant += progress[i];
  }
  accumulant = accumulant / ((float)(end - begin));
  return accumulant;
}

void Controller::print() {
  FILE* outputFilePointer = fopen("gap.dat", "w");
  if (outputFilePointer == NULL) {
    printf("Can't write %s\n", "gap.dat");
    exit(1);
  }
  for(vector<float>::iterator i = progress.begin(); i != progress.end(); i++) {
    fprintf(outputFilePointer, "%f ", *i);
  }
  fprintf(outputFilePointer, "\n");
  fclose(outputFilePointer);

  outputFilePointer = fopen("method.dat", "w");
  if (outputFilePointer == NULL) {
    printf("Can't write %s\n", "method.dat");
    exit(1);
  }
  for(vector<int>::iterator i = method.begin(); i != method.end(); i++) {
    fprintf(outputFilePointer, "%d ", *i);
  }
  fprintf(outputFilePointer, "\n");
  fclose(outputFilePointer);
}

template<int Kernel>
void initializeArrays(float* devData, int devDataPitchInFloats, float* devCache, int devCachePitchInFloats, int nPoints, int nDimension, float parameterA, float parameterB, float parameterC, float* devKernelDiag, float* devAlpha, float* devF, float* devLabels, int nthreads) { 
	//#pragma omp parallel for num_threads(nthreads)
	for (int index = 0; index < nPoints; index++) {
    if (Kernel == NEWGAUSSIAN) {
			devKernelDiag[index] = 1.0f;
		}
		if (Kernel == NEWPRECOMPUTED) {
			devKernelDiag[index] = devCache[index* devCachePitchInFloats + index];
		}
		devF[index] = -devLabels[index];
		devAlpha[index] = 0;
	}
}

void launchInitialization(float* devData, int devDataPitchInFloats, float* devCache, int devCachePitchInFloats, int nPoints, int nDimension, int kType, float parameterA, float parameterB, float parameterC, float* devKernelDiag, float* devAlpha, float* devF, float* devLabels, int nthreads) {
  switch (kType) {
  case NEWGAUSSIAN:
    initializeArrays<NEWGAUSSIAN>(devData, devDataPitchInFloats, devCache, devCachePitchInFloats, nPoints, nDimension, parameterA, parameterB, parameterC, devKernelDiag, devAlpha, devF, devLabels, nthreads);
    break;  
  case NEWPRECOMPUTED:
    initializeArrays<NEWPRECOMPUTED>(devData, devDataPitchInFloats, devCache, devCachePitchInFloats, nPoints, nDimension, parameterA, parameterB, parameterC, devKernelDiag, devAlpha, devF, devLabels, nthreads);
    break;  
  default:
    printf("Unsupported kernel \n");
    exit(1);
  }
}

template<int Kernel>
void takeFirstStep(void* devResult, float* devKernelDiag, float* devData, int devDataPitchInFloats, float* devCache, int devCachePitchInFloats, float* devAlpha, float cost, int nDimension, int iLow, int iHigh, float parameterA, float parameterB, float parameterC) { 
                                     
  float eta = devKernelDiag[iHigh] + devKernelDiag[iLow];
  float phiAB;

  if (Kernel == NEWGAUSSIAN) {
    float* pointerA = devData + iHigh;
    float* pointerB = devData + iLow;
    float acc = 0.0f;
    for (int i = 0; i < nDimension; i++) {
      float diff = pointerA[i*devDataPitchInFloats] - pointerB[i*devDataPitchInFloats];
      acc += diff*diff;
    }
    phiAB = exp(parameterA * acc);
  } 
  if (Kernel == NEWPRECOMPUTED) {
    phiAB = devCache[iHigh * devCachePitchInFloats + iLow];
  }

	eta = eta - 2*phiAB;
	//For the first step, we know alpha1Old == alpha2Old == 0, and we know sign == -1

	//And we know eta > 0
	float alpha2New = 2/eta; //Just boil down the algebra
	if (alpha2New > cost) {
		alpha2New = cost;
	}
	devAlpha[iLow] = alpha2New;
	devAlpha[iHigh] = alpha2New;
	
	*((float*)devResult + 0) = 0.0;
	*((float*)devResult + 1) = 0.0;
	*((float*)devResult + 2) = 1.0;
	*((float*)devResult + 3) = -1.0;
	*((float*)devResult + 6) = alpha2New;
	*((float*)devResult + 7) = alpha2New;
}

void launchTakeFirstStep(void* devResult, float* devKernelDiag, float* devData, int devDataPitchInFloats, float* devCache, int devCachePitchInFloats, float* devAlpha, float cost, int nDimension, int iLow, int iHigh, int kType, float parameterA, float parameterB, float parameterC, int nthreads) {
  switch (kType) {
  case NEWGAUSSIAN:
    takeFirstStep<NEWGAUSSIAN>(devResult, devKernelDiag, devData, devDataPitchInFloats, devCache, devCachePitchInFloats, devAlpha, cost, nDimension, iLow, iHigh, parameterA, parameterB, parameterC);
    break;  
  case NEWPRECOMPUTED:
    takeFirstStep<NEWPRECOMPUTED>(devResult, devKernelDiag, devData, devDataPitchInFloats, devCache, devCachePitchInFloats, devAlpha, cost, nDimension, iLow, iHigh, parameterA, parameterB, parameterC);
    break;  
  default:
    printf("Unsupported kernel \n");
    exit(1);
  }
}

void performClassification(float *data, int nData, float *supportVectors, int nSV, int nDimension, float* alpha, Kernel_params* kp, float** p_result)
{
	int total_nPoints = nData;
	int nPoints;	
	float gamma,coef0,b;
	int degree;
	KernelType kType;
	
	if(kp->kernel_type.compare(0,3,"rbf") == 0)
	{
		//printf("Found RBF kernel\n");
		gamma=kp->gamma;
		b=kp->b;
		kType = NEWGAUSSIAN;
	}
	else if(kp->kernel_type.compare(0,6,"linear") == 0)
	{
		//printf("Found linear kernel\n");
		gamma = 1.0;
		b=kp->b;
		kType = NEWLINEAR;
	}
	else if(kp->kernel_type.compare(0,11,"precomputed") == 0)
	{
		//printf("Found precomputed kernel\n");
		b=kp->b;
		kType = NEWPRECOMPUTED;
	}
	else
	{
		printf("Error: Unknown kernel type - %s\n",kp->kernel_type.c_str());
		exit(0);
	}
	
	int nBlocksSV = intDivideRoundUp(nSV,BLOCKSIZE);

	float* devSV;
	size_t devSVPitch;
	devSV = supportVectors;
	int devSVPitchInFloats = nDimension;

	float* devAlpha;
	devAlpha = alpha;

	float* devLocalValue;
	float* devResult;
	float* result = (float*)malloc(total_nPoints*sizeof(float));
	*(p_result) = result;

	float* devSVDots;
	devSVDots = new float[nSV];

	size_t free_memory,total_memory;
	free_memory = 1*1024*1024*1024; //1GB 
	size_t free_memory_floats = free_memory/sizeof(float);
	
	nPoints = ((free_memory_floats-devSVPitchInFloats*nDimension-nSV-nSV)/(nDimension+1+devSVPitchInFloats+1+nBlocksSV));
	nPoints = (nPoints>>7)<<7;		//for pitch limitations assigning to be a multiple of 128
	
	nPoints = (nPoints < total_nPoints)?(nPoints):(total_nPoints); //for few points
	nPoints = (nPoints < (int)MAX_POINTS)?(nPoints):((int)MAX_POINTS); //for too many points	

	float* devData;
	size_t devDataPitch;
	//devData = new float[nPoints*nDimension];
	devDataPitch = nDimension*sizeof(float);

	int devDataPitchInFloats = ((int)devDataPitch) / sizeof(float);

	float* devDataDots;
	devDataDots = new float[nPoints];
	
	float* devDots;
	size_t devDotsPitch;
	devDots = new float[nSV*nPoints];
	devDotsPitch = sizeof(float)*nSV;

	if(kType == NEWPRECOMPUTED) {
		devData = data;
    //if (vid==9053 && fold==1) {printf("%d\n", devDataPitchInFloats); exit(1);}
		//#pragma omp parallel for
		for (int i = 0; i < nPoints; i++) {
			float s = b;
			float* __restrict__ ptr = devData + i*devDataPitchInFloats;
			for (int j = 0; j < nSV; j++) {
				int index = (int)devSV[j];
				s += devAlpha[j]*ptr[index];
			}
			//devResult[i] = s;
      result[i] = s;
      //printf("%f ", s);
		}
	} else {
    devResult = new float[nPoints];
	  if(kType == NEWGAUSSIAN)
	  {
		  makeSelfDots(devSV, devSVPitchInFloats, devSVDots, nSV, nDimension);
	  }

	  int iteration=1;
	  for(int dataoffset=0; dataoffset<total_nPoints; dataoffset += nPoints) 
	  {
		  // code for copying data
  		if(dataoffset+nPoints > total_nPoints)
	  	{
	  		nPoints = total_nPoints-dataoffset;
	  	}
		
	  	//FIX PROBLEM HERE --- SEND DATA OR TRANSPOSED DATA???
	  	//Decided to use transposed data only..
	  	devData = data + dataoffset*devDataPitchInFloats;
		int devDotsPitchInFloats = ((int)devDotsPitch)/ sizeof(float);
	
  		if(kType == NEWGAUSSIAN)
	  	{
	  		makeSelfDots(devData, devDataPitchInFloats, devDataDots, nPoints, nDimension);
	  		memset(devDots, 0, sizeof(float)*devDotsPitchInFloats*nPoints);
	  		makeDots(devDots, devDotsPitchInFloats, devSVDots, devDataDots, nSV, nPoints);
	  	}
  		float sgemmAlpha, sgemmBeta;
  		if(kType == NEWGAUSSIAN)
  		{
  			sgemmAlpha = 2*gamma;
  			sgemmBeta = -gamma;
  		}
  		else
  		{
  			sgemmAlpha = gamma;
  			sgemmBeta = 0.0f;
  		}
  		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nPoints, nSV, nDimension, sgemmAlpha, devData, devDataPitchInFloats, devSV, devSVPitchInFloats, sgemmBeta, devDots, devDotsPitchInFloats); //for rowmajor formats
  		computeKernels(devDots, devDotsPitchInFloats, devAlpha, nPoints, nSV, kType, coef0, degree, b, devResult);
  		memcpy(result+dataoffset, devResult, nPoints*sizeof(float));
  		iteration++;
  	}
    //delete [] devData;
	}//non-precomp kernel
	delete [] devSVDots;
	delete [] devDataDots;
	delete [] devDots;
}

void makeSelfDots(float* devSource, int devSourcePitchInFloats, float* devDest, int sourceCount, int sourceLength) {
	//#pragma omp parallel for
	for (int i = 0; i < sourceCount; i++) {
		float dot = 0.0f;
		float* __restrict__ ptr = devSource + i*devSourcePitchInFloats;
		#pragma simd	
		for (int j = 0; j < sourceLength; j++) {
			dot += ptr[j]*ptr[j];			
		}
		devDest[i] = dot;
	}
}

void makeDots(float* devDots, int devDotsPitchInFloats, float* devSVDots, float* devDataDots, int nSV, int nPoints) {
	//#pragma omp parallel for
	for (int j = 0; j < nPoints; j++) {
		#pragma simd
		for (int i = 0; i < nSV; i++) {
			devDots[j*devDotsPitchInFloats + i] = devSVDots[i] + devDataDots[j];
		}
	}
}

float kernel(const float v, const float coef0, const int degree, const KernelType kType) {
	float res = 0.0f;
	switch(kType) {
	case NEWGAUSSIAN:
		res = exp(v); break;
	default: //also, linear and precomputed kernels
		res = v;
	}

	return res;
}

void computeKernels(float* devNorms, int devNormsPitchInFloats, float* devAlpha, int nPoints, int nSV, const KernelType kType, float coef0, int degree, float b, float* devResult) {

	//#pragma omp parallel for
	for (int j = 0; j < nPoints; j++) {
		float res = 0.0f;
		#pragma simd
		for (int i = 0; i < nSV; i++) {
			res += devAlpha[i]*kernel(devNorms[j*devNormsPitchInFloats + i], coef0, degree, kType);
		}
		devResult[j] = res + b;
	}
}
