/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include <mpi.h>
#include "Marginal_Scheduler.h"
#include "common.h"
#include "MatComputation.h"
#include "CorrMatAnalysis.h"
#include "Classification.h"
#include "Preprocessing.h"
#include "FileProcessing.h"
#include "SVMClassification.h"
#include "FisherScoring.h"
#include "ErrorHandling.h"

// marginal screening computation only needs one mask file
void Marginal_Scheduler(int me, int nprocs, int step, RawMatrix** r_matrices, int taskType, Trial* trials, int nTrials, int nHolds, int nSubs, int nFolds, const char* output_file, const char* mask_file)
{
  int i;
  RawMatrix** masked_matrices=NULL;
#ifndef __MIC__
  if (me == 0)
  {
    if (mask_file!=NULL)
    {
      masked_matrices = GetMaskedMatrices(r_matrices, nSubs, mask_file);
      //MatrixPermutation(masked_matrices1, nSubs);
    }
    else
      masked_matrices = r_matrices;
  }
#endif
  double tstart = MPI_Wtime();
  if (me != 0)
  {
    masked_matrices = new RawMatrix*[nSubs];
    for (i=0; i<nSubs; i++)
    {
      masked_matrices[i] = new RawMatrix();
    }
  }
  for (i=0; i<nSubs; i++)
  {
    MPI_Bcast((void*)masked_matrices[i], sizeof(RawMatrix), MPI_CHAR, 0, MPI_COMM_WORLD);
  }
  int row = masked_matrices[0]->row;
  int col = masked_matrices[0]->col;
  if (me != 0)
  {
    for (i=0; i<nSubs; i++)
    {
      masked_matrices[i]->matrix = new float[row*col];
    }
  }
  for (i=0; i<nSubs; i++)
  {
    MPI_Bcast((void*)(masked_matrices[i]->matrix), row*col, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double tstop = MPI_Wtime();
  int nBlocksPerSub = nTrials / nSubs;  // assume each subject has the same number of blocks
  // prepare the data, theoretically we can only broadcast those data
  int nTrialsUsed = nTrials-nHolds;
  float* data = prepare_data(masked_matrices, trials, row, nTrialsUsed, nBlocksPerSub);
  // get label information
  int* labels = GatherLabels(trials, nTrials);
  if (me == 0)
  {
#ifndef __MIC__
    cout.precision(6);
    cout<<"data transferring time: "<<tstop-tstart<<"s"<<endl;
    cout<<"#voxels for mask: "<<row<<endl;
    VoxelScore* scores = compute_first_order(masked_matrices, trials, nTrials, nHolds, nBlocksPerSub, data, labels);
    write_result(output_file, 1, scores, row, mask_file);
    cout<<"first order done!"<<endl;
    // allocate second order matrix
    cout<<"starting second order:"<<endl;
    float* second_order = new float[row*row];
    Do_Marginal_Master(nprocs, step, row, second_order, output_file, mask_file);
    int topK = 1000;
    VoxelScore* marginalLikelihood = compute_marginal_info(scores, topK, row, second_order);
    write_result(output_file, 2, marginalLikelihood, row, mask_file);
    // writing the second order matrix
    /*char fullfilename[MAXFILENAMELENGTH];
    sprintf(fullfilename, "%s", output_file);
    strcat(fullfilename, "_second_order.bin");
    FILE* fp=fopen(fullfilename, "wb");
    fwrite((const void*)second_order, sizeof(float), row*row, fp);
    fclose(fp);*/
    delete [] second_order; // bds []
#endif
  }
  else
  {
    Do_Marginal_Slave(me, 0, data, labels, row, taskType, nTrialsUsed);  // 0 for master id
  }
}

// function for sort
bool cmp1(VoxelScore w1, VoxelScore w2)
{
  return w1.score>w2.score;
}


// finally might need to move those functions that are not related to scheduling to somewhere else
float* prepare_data(RawMatrix** masked_matrices, Trial* trials, int row, int nTrialsUsed, int nBlocksPerSub)
{
  float* data = new float[row*nTrialsUsed];
  int i;
  // Get averaged data for each block
  #pragma omp parallel for private(i)
  for (i=0; i<nTrialsUsed; i++)
  {
    int sid = trials[i].sid;
    int sc = trials[i].sc;
    int ec = trials[i].ec;
    for (int j=0; j<row; j++)
    {
      int col = masked_matrices[sid]->col;
      float mean_value = 0.0;
      for (int k=sc; k<=ec; k++)
      {
        mean_value += masked_matrices[sid]->matrix[j*col+k];
      }
      mean_value /= (ec-sc+1);
      data[j*nTrialsUsed+i] = mean_value;
    }
  }
  // z-score data or convert the data to be the same L2 norm
  #pragma omp parallel for private(i)
  for (i=0; i<row; i++)
  {
    //z_score(data+i*nTrialsUsed, nTrialsUsed); // z-score over all blocks
    /*for (int j=0; j<nTrialsUsed; j+=nBlocksPerSub)  // z-score within subjects
    {
      z_score(data+i*nTrialsUsed+j, nBlocksPerSub);
    }*/
    double l2norm = 0.0;
    for (int j=0; j<nTrialsUsed; j++) l2norm+=data[i*nTrialsUsed+j]*data[i*nTrialsUsed+j];
    l2norm = sqrt(l2norm);
    for (int j=0; j<nTrialsUsed; j++) data[i*nTrialsUsed+j] = data[i*nTrialsUsed+j] / l2norm;
  }
  return data;
}

VoxelScore* compute_first_order(RawMatrix** masked_matrices, Trial* trials, int nTrials, int nHolds, int nBlocksPerSub, float* data, int* labels)
{
  int nTrialsUsed = nTrials-nHolds;
  int row = masked_matrices[0]->row;
  int i;
  VoxelScore* scores = new VoxelScore[row];
  #pragma omp parallel for private(i)
  for (i=0; i<row; i++)
  {
    scores[i].vid=i;
    scores[i].score=DoIteration(data+i*nTrialsUsed, NULL, nTrialsUsed, labels, LOGISTICTRHEHOLD, 2, i, -1);
  }
  sort(scores, scores+row, cmp1);
  return scores;
}

void compute_second_order(float* data, int* labels, int nTrialsUsed, int row, float* second_order, int sr, int step)
{
  //omp_lock_t writelock;
  //omp_init_lock(&writelock);
  // int n=0;
  int i=0;
  #pragma omp parallel for private(i) schedule(dynamic)
  for (i=0; i<step; i++)
  {
    /*omp_set_lock(&writelock);
    n++;
    printf("%d ", n);
    fflush(stdout);
    omp_unset_lock(&writelock);*/
    int j;
    for (j=0; j<row; j++)
    {
      if ((i+sr)!=j)
      {
        second_order[i*row+j]=DoIteration(data+(i+sr)*nTrialsUsed, data+j*nTrialsUsed, nTrialsUsed, labels, LOGISTICTRHEHOLD, 3, i+sr, j);
      }
      else
      {
        second_order[i*row+j]=-9999.0;
      }
    }
  }
}

int* GatherLabels(Trial* trials, int nTrials)
{
  int i;
  int* labels = (int*)_mm_malloc(sizeof(int)*nTrials, 64);  // malloc with memory alignment
  #pragma omp parallel for
  #pragma vector aligned
  for (i=0; i<nTrials; i++)
  {
    labels[i]=trials[i].label;
  }
  return labels;
}

// compute log likelihood gains
VoxelScore* compute_marginal_info(VoxelScore* scores, int topK, int row, float* second_order)
{
  int i;
  bool isTopK[row];
  memset((void*)isTopK, false, row*sizeof(bool));
  for (i=0; i<topK; i++)
  {
    isTopK[scores[i].vid]=true;
  }
  VoxelScore* marginalLikelihood = new VoxelScore[row];
  //#pragma omp parallel for private(i)
  for (i=0; i<row; i++)
  {
    marginalLikelihood[i].vid=i;
    marginalLikelihood[i].score=0.0;
    if (isTopK[i]) continue;
    for (int j=0; j<row; j++)
    {
      if (!isTopK[j]) continue;
      float cur_score = second_order[i*row+j]-scores[j].score;
      //if (marginalLikelihood[i].score<cur_score)
      //{
        marginalLikelihood[i].score += cur_score;
      //}
    }
  }
  sort(marginalLikelihood, marginalLikelihood+row, cmp1);
  return marginalLikelihood;
}

void write_result(const char* output_file, int order, VoxelScore* scores, int row, const char* mask_file)
{
#ifndef __MIC__
  char suffix1[MAXFILENAMELENGTH];
  char suffix2[MAXFILENAMELENGTH];
  if (order==1)
  {
    strcpy(suffix1, "_first_order_list.txt");   // doesn't need '\n'
    strcpy(suffix2, "_first_order_seq.nii.gz");
  }
  else if (order==2)
  {
    strcpy(suffix1, "_second_order_list.txt");   // doesn't need '\n'
    strcpy(suffix2, "_second_order_seq.nii.gz");
  }
  char fullfilename[MAXFILENAMELENGTH];
  sprintf(fullfilename, "%s", output_file);
  strcat(fullfilename, suffix1);
  ofstream ofile(fullfilename);
  for (int i=0; i<row; i++)
  {
    ofile<<scores[i].vid<<" "<<scores[i].score<<endl;
  }
  ofile.close();
  int* data_ids = (int*)GenerateNiiDataFromMask(mask_file, scores, row, DT_SIGNED_INT);
  sprintf(fullfilename, "%s", output_file);
  strcat(fullfilename, suffix2);
  WriteNiiGzData(fullfilename, mask_file, (void*)data_ids, DT_SIGNED_INT);
#endif
}

/* the master node splits the tasks and assign them to slave nodes. 
When a slave nodes return, the master node would send another tasks to it. 
The master node finally collects all results and sort them to write to a file. 
The mask file here is the sample nifti file for writing */
void Do_Marginal_Master(int nprocs, int step, int row, float* second_order, const char* output_file, const char* mask_file)
{
#ifndef __MIC__
  int curSr = 0;
  int i;
  int total = row / step;
  if (row%step != 0)
  {
    total += 1;
  }
  cout<<"total task: "<<total<<endl;
  int sentCount = 0;
  int doneCount = 0;
  int sendMsg[2];
  for (i=1; i<nprocs; i++)  // fill up the processes first
  {
    sendMsg[0] = curSr;
    sendMsg[1] = step;
    if (curSr+step>row) // overflow, so only get the left
    {
      sendMsg[1] = row - curSr;
    }
    curSr += sendMsg[1];
    MPI_Send(sendMsg,  /* message buffer, the correlation vector */
           2,                  /* number of data to send */
           MPI_INT,                       /* data item is float */
           i,                            /* destination process rank */
           COMPUTATIONTAG,                      /* user chosen message tag */
           MPI_COMM_WORLD);                 /* default communicator */
    sentCount++;
  }
  MPI_Status status;
  while (sentCount < total)
  {
    int recvMsg[2];
    int curPosition;
    int curStep;
    float elapse;
    // get the elapse time
    MPI_Recv(&elapse,      /* message buffer */
           1,              /* numbers of data to receive */
           MPI_FLOAT,          /* of type float real */
           MPI_ANY_SOURCE,                       /* receive from any sender */
           ELAPSETAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    // get the position  and the step length of the matrix
    MPI_Recv(recvMsg,      /* message buffer */
           2,              /* numbers of data to receive */
           MPI_INT,          /* of type float real */
           status.MPI_SOURCE,       /* receive from any sender */
           POSITIONTAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    curPosition = recvMsg[0];
    curStep = recvMsg[1];
    // get partial second order matrix
    MPI_Recv(second_order+curPosition*row,      /* message buffer */
           curStep*row,              /* numbers of data to receive */
           MPI_FLOAT,          /* of type float real */
           status.MPI_SOURCE,                       /* receive from the previous sender */
           SECONDORDERTAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    doneCount++;
    ///////////cout<<*(second_order+curPosition*row)<<" "<<*(second_order+curPosition*row+1)<<endl;
    cout.precision(4);
    cout<<doneCount<<'\t'<<elapse<<"s\t"<<flush;
    if (doneCount%6==0)
    {
      cout<<endl;
    }
    sendMsg[0] = curSr;
    sendMsg[1] = step;
    if (curSr+step>row) // overflow, so only get the left
    {
      sendMsg[1] = row - curSr;
    }
    curSr += sendMsg[1];
    MPI_Send(sendMsg,  /* message buffer, the correlation vector */
           2,                  /* number of data to send */
           MPI_INT,                       /* data item is float */
           status.MPI_SOURCE,             /* destination process rank */
           COMPUTATIONTAG,                      /* user chosen message tag */
           MPI_COMM_WORLD);                 /* default communicator */
    sentCount++;
  }
  while (doneCount < total)
  {
    int recvMsg[2];
    int curPosition;
    int curStep;
    float elapse;
    // get the elapse time
    MPI_Recv(&elapse,      /* message buffer */
           1,              /* numbers of data to receive */
           MPI_FLOAT,          /* of type float real */
           MPI_ANY_SOURCE,                       /* receive from any sender */
           ELAPSETAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    // get the position  and the step length of the matrix
    MPI_Recv(recvMsg,      /* message buffer */
           2,              /* numbers of data to receive */
           MPI_INT,          /* of type float real */
           status.MPI_SOURCE,       /* receive from any sender */
           POSITIONTAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    curPosition = recvMsg[0];
    curStep = recvMsg[1];
    // get partial second order matrix
    MPI_Recv(second_order+curPosition*row,      /* message buffer */
           curStep*row,              /* numbers of data to receive */
           MPI_FLOAT,          /* of type float real */
           status.MPI_SOURCE,                       /* receive from the previous sender */
           SECONDORDERTAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    doneCount++;
    cout.precision(4);
    cout<<doneCount<<'\t'<<elapse<<"s\t"<<flush;
    if (doneCount%6==0)
    {
      cout<<endl;
    }
  }
  for (i=1; i<nprocs; i++)  // tell all processes to stop
  {
    sendMsg[0] = -1;
    sendMsg[1] = -1;
    curSr += step;
    MPI_Send(sendMsg,  /* message buffer, the correlation vector */
           2,                  /* number of data to send */
           MPI_INT,                       /* data item is float */
           i,                            /* destination process rank */
           COMPUTATIONTAG,                      /* user chosen message tag */
           MPI_COMM_WORLD);                 /* default communicator */
  }
#endif
}

/* the slave node listens to the master node and does the task that the master node assigns. 
A task is a matrix multiplication to get the correlation vectors of some voxels. 
The slave node also does some preprocessing on the correlation vectors then 
analyzes the correlatrion vectors (either do classification or compute the average correlation coefficients.*/
void Do_Marginal_Slave(int me, int masterId, float* data, int* labels, int row, int taskType, int nTrialsUsed)
{
  int recvMsg[2];
  MPI_Status status;
  while (true)
  {
    MPI_Recv(recvMsg,      /* message buffer */
           2,              /* numbers of data to receive */
           MPI_INT,          /* of type float real */
           masterId,                       /* receive from any sender */
           COMPUTATIONTAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    double tstart = MPI_Wtime();
    int sr = recvMsg[0];
    int step = recvMsg[1];
    if (sr == -1) // finish flag
    {
      break;
    }
    float* second_order = new float[row*step];
    if (taskType==9)
    {
      compute_second_order(data, labels, nTrialsUsed, row, second_order, sr, step);
    }
    else
    {
      FATAL("wrong task type in marginal screening!");
    }
    double tstop = MPI_Wtime();
    float elapse = float(tstop-tstart);
    MPI_Send(&elapse,  /* message buffer, the correlation vector */
           1,                  /* number of data to send */
           MPI_FLOAT,                       /* data item is float */
           masterId,                            /* destination process rank */
           ELAPSETAG,                      /* user chosen message tag */
           MPI_COMM_WORLD);                 /* default communicator */
    MPI_Send(recvMsg,  /* message buffer, the correlation vector */
           2,                  /* number of data to send */
           MPI_INT,                       /* data item is float */
           masterId,                            /* destination process rank */
           POSITIONTAG,                      /* user chosen message tag */
           MPI_COMM_WORLD);                 /* default communicator */
    MPI_Send(second_order,  /* message buffer, the correlation vector */
           row*step,                  /* number of data to send */
           MPI_FLOAT,                       /* data item is float */
           masterId,                            /* destination process rank */
           SECONDORDERTAG,                      /* user chosen message tag */
           MPI_COMM_WORLD);                 /* default communicator */
    delete [] second_order; // bds []
  }
}
