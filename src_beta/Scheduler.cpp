/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include <mpi.h>
#include "Scheduler.h"
#include "common.h"
#include "MatComputation.h"
#include "CorrMatAnalysis.h"
#include "Classification.h"
#include "Preprocessing.h"
#include "FileProcessing.h"
#include "SVMClassification.h"
#include "VoxelwiseAnalysis.h"
#include "ErrorHandling.h"

// two mask files can be different
void Scheduler(int me, int nprocs, int step, RawMatrix** r_matrices, RawMatrix** r_matrices2, int taskType, Trial* trials, int nTrials, int nHolds, int nSubs, int nFolds, const char* output_file, const char* mask_file1, const char* mask_file2, int shuffle, const char* permute_book_file)
{
  int i;
  RawMatrix** masked_matrices1=NULL;
  RawMatrix** masked_matrices2=NULL;
#ifndef __MIC__
  if (me == 0)
  {
    if (mask_file1!=NULL)
    {
      masked_matrices1 = GetMaskedMatrices(r_matrices, nSubs, mask_file1);
    }
    else
      masked_matrices1 = r_matrices;
    if (mask_file2!=NULL)
    {
      masked_matrices2 = GetMaskedMatrices(r_matrices2, nSubs, mask_file2);
    }
    else
    {
      masked_matrices2 = r_matrices2;
    }
    if (shuffle==1 || shuffle==2)
    {
      unsigned int seed = (unsigned int)time(NULL);
      MatrixPermutation(masked_matrices1, nSubs, seed, permute_book_file);
      MatrixPermutation(masked_matrices2, nSubs, seed, permute_book_file);
    }
#ifdef __USE_MIC__  // currently, only for MIC jobs we preprocess the activity matrices to save bandwidth and onboard normalization time
    if (taskType==0)
    {
      Trial* buf_trials = new Trial[nTrials];
      memcpy((void*)buf_trials, (const void*)trials, sizeof(Trial)*nTrials);
      PreprocessMatrices(masked_matrices1, buf_trials, trials, nSubs, nTrials);
      PreprocessMatrices(masked_matrices2, buf_trials, trials, nSubs, nTrials);
      delete buf_trials;
    }
#endif
  }
#endif
  double tstart = MPI_Wtime();
#ifdef __USE_MIC__
  // resend trials because it may change in PreprocessMatrices  
  if (taskType==0)
  {
    MPI_Bcast((void*)trials, sizeof(Trial)*nTrials, MPI_CHAR, 0, MPI_COMM_WORLD);
  }
#endif
  if (me != 0)
  {
    masked_matrices1 = new RawMatrix*[nSubs];
    masked_matrices2 = new RawMatrix*[nSubs];
    for (i=0; i<nSubs; i++)
    {
      masked_matrices1[i] = new RawMatrix();
      masked_matrices2[i] = new RawMatrix();
    }
  }
  for (i=0; i<nSubs; i++)
  {
    MPI_Bcast((void*)masked_matrices1[i], sizeof(RawMatrix), MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*)masked_matrices2[i], sizeof(RawMatrix), MPI_CHAR, 0, MPI_COMM_WORLD);
  }
  int r1 = masked_matrices1[0]->row;
  int c1 = masked_matrices1[0]->col;
  int r2 = masked_matrices2[0]->row;
  int c2 = masked_matrices2[0]->col;
  if (me != 0)
  {
    for (i=0; i<nSubs; i++)
    {
      masked_matrices1[i]->matrix = new float[r1*c1];
      masked_matrices2[i]->matrix = new float[r2*c2];
    }
  }
  for (i=0; i<nSubs; i++)
  {
    MPI_Bcast((void*)(masked_matrices1[i]->matrix), r1*c1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*)(masked_matrices2[i]->matrix), r2*c2, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double tstop = MPI_Wtime();
  if (me == 0)
  {
    cout.precision(6);
    cout<<"data transferring time: "<<tstop-tstart<<"s"<<endl;
    cout<<"#voxels for mask 1: "<<r1<<endl;
    cout<<"#voxels for mask 2: "<<r2<<endl;
    DoMaster(nprocs, step, r1, output_file, mask_file1);
  }
  else
  {
    //int preset_step=30;
    DoSlave(me, 0, masked_matrices1, masked_matrices2, taskType, trials, nTrials, nHolds, nSubs, nFolds, step);  // 0 for master id
  }
}

// function for sort
bool cmp(VoxelScore w1, VoxelScore w2)
{
  return w1.score>w2.score;
}

/* the master node splits the tasks and assign them to slave nodes. 
When a slave nodes return, the master node would send another tasks to it. 
The master node finally collects all results and sort them to write to a file. 
The mask file here is the sample nifti file for writing */
void DoMaster(int nprocs, int step, int row, const char* output_file, const char* mask_file)
{
#ifndef __MIC__
  int curSr = 0;
  int i, j;
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
  VoxelScore* scores = new VoxelScore[row];  // one voxel one score
  int totalLength = 0;
  MPI_Status status;
  while (sentCount < total)
  {
    int curLength;
    float elapse;
    // get the elapse time
    MPI_Recv(&elapse,      /* message buffer */
           1,              /* numbers of data to receive */
           MPI_FLOAT,          /* of type float real */
           MPI_ANY_SOURCE,                       /* receive from any sender */
           ELAPSETAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    // get the length of message first
    MPI_Recv(&curLength,      /* message buffer */
           1,              /* numbers of data to receive */
           MPI_INT,          /* of type float real */
           status.MPI_SOURCE,       /* receive from any sender */
           LENGTHTAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    // get the classifier array
    MPI_Recv(scores+totalLength,      /* message buffer */
           curLength*2,              /* numbers of data to receive */
           MPI_FLOAT,          /* of type float real */
           status.MPI_SOURCE,                       /* receive from the previous sender */
           VOXELCLASSIFIERTAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    totalLength += curLength;
    doneCount++;
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
    int curLength;
    float elapse;
    // get the elapse time
    MPI_Recv(&elapse,      /* message buffer */
           1,              /* numbers of data to receive */
           MPI_FLOAT,          /* of type float real */
           MPI_ANY_SOURCE,                       /* receive from any sender */
           ELAPSETAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    // get the length of message first
    MPI_Recv(&curLength,      /* message buffer */
           1,              /* numbers of data to receive */
           MPI_INT,          /* of type float real */
           status.MPI_SOURCE,      /* receive from any sender */
           LENGTHTAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    // get the classifier array
    MPI_Recv(scores+totalLength,      /* message buffer */
           curLength*2,              /* numbers of data to receive */
           MPI_FLOAT,          /* of type float real */
           status.MPI_SOURCE,                       /* receive from any sender */
           VOXELCLASSIFIERTAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    totalLength += curLength;
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
  sort(scores, scores+totalLength, cmp);
  cout<<"Total length: "<<totalLength<<endl;
  // deal with output_file here////////////////////////////////////////////
  char fullfilename[MAXFILENAMELENGTH];
  sprintf(fullfilename, "%s", output_file);
  strcat(fullfilename, "_list.txt");
  ofstream ofile(fullfilename);
  for (j=0; j<totalLength; j++)
  {
    ofile<<scores[j].vid<<" "<<scores[j].score<<endl;
  }
  ofile.close();
  int* data_ids = (int*)GenerateNiiDataFromMask(mask_file, scores, totalLength, DT_SIGNED_INT);
  sprintf(fullfilename, "%s", output_file);
  strcat(fullfilename, "_seq.nii.gz");
  WriteNiiGzData(fullfilename, mask_file, (void*)data_ids, DT_SIGNED_INT);
  float* data_scores = (float*)GenerateNiiDataFromMask(mask_file, scores, totalLength, DT_FLOAT32);
  sprintf(fullfilename, "%s", output_file);
  strcat(fullfilename, "_score.nii.gz");
  WriteNiiGzData(fullfilename, mask_file, (void*)data_scores, DT_FLOAT32);
#endif
}

/* the slave node listens to the master node and does the task that the master node assigns. 
A task is a matrix multiplication to get the correlation vectors of some voxels. 
The slave node also does some preprocessing on the correlation vectors then 
analyzes the correlatrion vectors (either do classification or compute the average correlation coefficients.*/
void DoSlave(int me, int masterId, RawMatrix** matrices1, RawMatrix** matrices2, int taskType, Trial* trials, int nTrials, int nHolds, int nSubs, int nFolds, int preset_step)
{
  int recvMsg[2];
  MPI_Status status;
  int nVoxels = matrices2[0]->row;
#ifdef __MIC__
  Voxel* voxels = new Voxel();
  voxels->nTrials = nTrials;
  voxels->nVoxels = nVoxels;
  voxels->vid=new int[preset_step];
  voxels->corr_vecs = (float*)_mm_malloc(sizeof(float)*nTrials*nVoxels*preset_step, 64);  // close to the uint limit
  int ml = trials[0].ec;
  int row1=matrices1[0]->row;
  int row2=matrices2[0]->row;
  float* bufs1=new float[ml*row1*nTrials];
  float* bufs2=new float[ml*row2*nTrials];
  /*if (taskType==0)
  {
    float* svm_data[240];
    for (int i=0; i<240; i++)
    {
      svm_data[i] = new float[nTrials-nHolds];
    }
  }*/
  for (int i=0; i<nTrials; i++)
  {
    int cur_col = trials[i].sc;
    int sid = trials[i].sid;
    memcpy((void*)(bufs1+i*ml*row1), (const void*)(matrices1[sid]->matrix+cur_col*row1), sizeof(float)*ml*row1);
    memcpy((void*)(bufs2+i*ml*row2), (const void*)(matrices2[sid]->matrix+cur_col*row2), sizeof(float)*ml*row2);
  }
  for (int i=0; i<nSubs; i++)
  {
    delete matrices1[i]->matrix;
    delete matrices2[i]->matrix;
  }
#endif
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
    int unit_step = preset_step;
    int cur_step = 0;
    /*while (cur_step < step)
    {
      int cur_sr = sr+cur_step;
      unit_step = step-cur_step>unit_step?unit_step:step-cur_step;
      cur_step += unit_step;*/
#ifdef __MIC__
#if __MEASURE_TIME__
      double t1 = MPI_Wtime();
#endif
      voxels = ComputeAllVoxelsAnalysisData(voxels, trials, nTrials, nSubs, sr, step, matrices1, matrices2, bufs1, bufs2);
#if __MEASURE_TIME__
      double t2 = MPI_Wtime();
      cout<<"voxel computing and normalization: "<<t2-t1<<"s"<<endl<<flush;
#endif
#else
      CorrMatrix** c_matrices;
#if __MEASURE_TIME__
      double t1 = MPI_Wtime();
#endif
      c_matrices = ComputeAllTrialsCorrMatrices(trials, nTrials, sr, step, matrices1, matrices2);
#if __MEASURE_TIME__
      double t2 = MPI_Wtime();
      cout<<"matrix computing: "<<t2-t1<<"s"<<endl;
#endif
#endif
      VoxelScore* scores = NULL;
      switch (taskType)
      {
        case 0:
#ifdef __MIC__
#if __MEASURE_TIME__
          //t1 = MPI_Wtime();
#endif
          //PreprocessAllVoxelsAnalysisData_flat(voxels, step, nSubs);
          //PreprocessAllVoxelsAnalysisData(voxels, unit_step, nSubs);
#if __MEASURE_TIME__
          t2 = MPI_Wtime();
          //cout<<"preprocessing: "<<t2-t1<<"s"<<endl;
#endif
          //scores = GetVoxelwiseSVMPerformance(me, trials, voxels, unit_step, nTrials-nHolds, nFolds);
          scores = GetVoxelwiseNewSVMPerformance(me, trials, voxels, unit_step, nTrials-nHolds, nFolds);
          //cout<<total_count<<endl;
          //total_count=0;
#if __MEASURE_TIME__
          t1 = MPI_Wtime();
          cout<<"svm processing: "<<t1-t2<<"s"<<endl;
#endif
#else
          // Fisher transform and z-score (across the blocks) the data here
#if __MEASURE_TIME__
          t1 = MPI_Wtime();
#endif
          corrMatPreprocessing(c_matrices, nTrials, nSubs);
#if __MEASURE_TIME__
          t2 = MPI_Wtime();
          cout<<"matrix normalization: "<<t2-t1<<"s"<<endl;
#endif
          scores = GetSVMPerformance(me, c_matrices, nTrials-nHolds, nFolds);
#if __MEASURE_TIME__
          t1 = MPI_Wtime();
          cout<<"svm processing: "<<t1-t2<<"s"<<endl;
#endif
#endif
          break;
        case 1:
#ifdef __MIC__
          // TODO
#else
          // Fisher transform and z-score (across the blocks) the data here
          corrMatPreprocessing(c_matrices, nTrials, nSubs);
          scores = GetDistanceRatio(me, c_matrices, nTrials-nHolds);
#endif
          break;
        case 3:
#ifdef __MIC__
          //TODO
#else
          scores = GetCorrVecSum(me, c_matrices, nTrials);
#endif
          break;
        default:
          FATAL("unknown task type");
      }
    //}
    double tstop = MPI_Wtime();
    float elapse = float(tstop-tstart);
    //cout<<elapse<<endl<<flush;
    MPI_Send(&elapse,  /* message buffer, the correlation vector */
           1,                  /* number of data to send */
           MPI_FLOAT,                       /* data item is float */
           masterId,                            /* destination process rank */
           ELAPSETAG,                      /* user chosen message tag */
           MPI_COMM_WORLD);                 /* default communicator */
    MPI_Send(&step,  /* message buffer, the correlation vector */
           1,                  /* number of data to send */
           MPI_INT,                       /* data item is float */
           masterId,                            /* destination process rank */
           LENGTHTAG,                      /* user chosen message tag */
           MPI_COMM_WORLD);                 /* default communicator */
    MPI_Send(scores,  /* message buffer, the correlation vector */
           step*2,                  /* number of data to send */
           MPI_FLOAT,                       /* data item is float */
           masterId,                            /* destination process rank */
           VOXELCLASSIFIERTAG,                      /* user chosen message tag */
           MPI_COMM_WORLD);                 /* default communicator */
    delete scores;
#ifdef __MIC__
#else
    for (int i=0; i<nTrials; i++)
    {
      delete c_matrices[i]->matrix;
    }
    delete c_matrices;
#endif
  }
#ifdef __MIC__
  /*for (int i=0; i<preset_step; i++)
  {
    _mm_free(voxels[i]->corr_vecs);
  }*/
  _mm_free(voxels->corr_vecs);
  delete voxels->vid;
  delete voxels;
  /*for (int i=0; i<240; i++)
  {
    delete svm_data[i];
  }*/
#endif
}
