/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include <mpi.h>
#include "Scheduler.h"
#include "common.h"
#include "MatComputation.h"
#include "CustomizedMatrixMultiply.h"
#include "CorrMatAnalysis.h"
#include "Classification.h"
#include "Preprocessing.h"
#include "FileProcessing.h"
#include "SVMClassification.h"
#include "VoxelwiseAnalysis.h"
#include "ErrorHandling.h"

// two mask files can be different
void Scheduler(int me, int nprocs, int step, RawMatrix** r_matrices,
               RawMatrix** r_matrices2, Task taskType, Trial* trials,
               int nTrials, int nHolds, int nSubs, int nFolds,
               const char* output_file, const char* mask_file1,
               const char* mask_file2, int shuffle,
               const char* permute_book_file) {
  double tstart, tstop;
  RawMatrix** masked_matrices1 = NULL;
  RawMatrix** masked_matrices2 = NULL;
  TrialData* td1 = NULL;
  TrialData* td2 = NULL;

  using std::cout;
  using std::endl;

#ifndef __MIC__
  if (me == 0) {
    //WaitForDebugAttach();
    tstart = MPI_Wtime();
    if (mask_file1 != NULL) {
      if (r_matrices!=r_matrices2) {
        // after GetMaskedMatrices, the data stored in r_matrices have been deleted
        masked_matrices1 = GetMaskedMatrices(r_matrices, nSubs, mask_file1, true);
      }
      else if (mask_file2!=NULL && !strcmp(mask_file1, mask_file2)) {  // masked_matrcies2 can reuse masked_matrices1
        masked_matrices1 = GetMaskedMatrices(r_matrices, nSubs, mask_file1, true);
      }
      else {  // same raw data, different masks
        masked_matrices1 = GetMaskedMatrices(r_matrices, nSubs, mask_file1, false);
      }
    }
    else
    {
      masked_matrices1 = r_matrices;
    }
    if (mask_file2 != NULL) {
      if (r_matrices==r_matrices2 && mask_file1==NULL) {  // same data, mask1 is null
        masked_matrices2 = GetMaskedMatrices(r_matrices2, nSubs, mask_file2, false);
      }
      // same data, different masks; or different data
      // if masks are different, GetMaskedMatrices is still needed even if r_matrices==r_matrices2
      else if (r_matrices!=r_matrices2 || strcmp(mask_file1, mask_file2)) {
        // after GetMaskedMatrices, the data stored in r_matrices have been deleted
        masked_matrices2 = GetMaskedMatrices(r_matrices2, nSubs, mask_file2, true);
      }
      else {
        masked_matrices2 = masked_matrices1;
      }
    }
    else {
      masked_matrices2 = r_matrices2;
    }
    if (shuffle == 1 || shuffle == 2) {
      unsigned int seed = (unsigned int)time(NULL);
      MatrixPermutation(masked_matrices1, nSubs, seed, permute_book_file);
      MatrixPermutation(masked_matrices2, nSubs, seed, permute_book_file);
    }
    td1 = PreprocessMatrices(masked_matrices1, trials, nSubs, nTrials);
    td2 = PreprocessMatrices(masked_matrices2, trials, nSubs, nTrials);
    tstop = MPI_Wtime();
    cout.precision(6);
    cout << "data mask applying time: " << tstop - tstart << "s" << endl;
  }
#endif

  tstart = MPI_Wtime();
  if (me != 0) {
    td1 = new TrialData(-1, -1);
    td2 = new TrialData(-1, -1);
  }
  MPI_Bcast((void*)td1, sizeof(TrialData), MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*)td2, sizeof(TrialData), MPI_CHAR, 0, MPI_COMM_WORLD);
  if (me != 0) {
    td1->trialLengths = new int[td1->nTrials];
    td2->trialLengths = new int[td2->nTrials];
    td1->scs = new int[td1->nTrials];
    td2->scs = new int[td2->nTrials];
    size_t dataSize = sizeof(float) * (size_t)td1->nCols * (size_t)td1->nVoxels;
    //td1->data = (float*)_mm_malloc(
    //    sizeof(float) * (size_t)td1->nCols * (size_t)td1->nVoxels, 64);
    td1->data = (float*)malloc(dataSize);
    assert(td1->data);
    //td2->data = (float*)_mm_malloc(
    //    sizeof(float) * (size_t)td2->nCols * (size_t)td2->nVoxels, 64);
    dataSize = sizeof(float) * (size_t)td2->nCols * (size_t)td2->nVoxels;
    td2->data = (float*)malloc(dataSize);
    assert(td2->data);
  }
  /*if (me==0) { //output for real-time paper
    FILE* fp = fopen("facesceneNorm_Acti.bin", "wb");
    fwrite ((const void*)td1->data, sizeof(float), (size_t)td1->nCols * (size_t)td1->nVoxels, fp);
    fclose(fp);
    exit(1);
  }*/
  MPI_Bcast((void*)(td1->trialLengths), td1->nTrials, MPI_INT, 0,
            MPI_COMM_WORLD);
  MPI_Bcast((void*)(td2->trialLengths), td2->nTrials, MPI_INT, 0,
            MPI_COMM_WORLD);
  MPI_Bcast((void*)(td1->scs), td1->nTrials, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*)(td2->scs), td2->nTrials, MPI_INT, 0, MPI_COMM_WORLD);

  // Since can be too much data for a single send (max 2^31 elements or 4GB)
  // send as a series of trials
  // (voxels*trs_per_trial per send , for nSubs*numblocks sends)
  float* d1ptr = td1->data;
  float* d2ptr = td2->data;
  for (int t = 0; t < nTrials; t++) {
    size_t dataChunk1 = td1->trialLengths[t] * (size_t)td1->nVoxels;
    assert(dataChunk1 < (size_t)INT_MAX);
    size_t dataChunk2 = td2->trialLengths[t] * (size_t)td2->nVoxels;
    assert(dataChunk2 < (size_t)INT_MAX);
    MPI_Bcast((void*)d1ptr, (int)dataChunk1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*)d2ptr, (int)dataChunk2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    d1ptr += dataChunk1;
    d2ptr += dataChunk2;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  tstop = MPI_Wtime();
  if (me == 0) {
    cout.precision(6);
    cout << "data transferring time: " << tstop - tstart << "s" << endl;
    cout << "#voxels for mask 1: " << masked_matrices1[0]->row << endl;
    cout << "#voxels for mask 2: " << masked_matrices2[0]->row << endl;
    DoMaster(nprocs, step, masked_matrices1[0]->row, output_file, mask_file1);
  } else {
    DoSlave(me, 0, td1, td2, taskType, trials, nTrials, nHolds, nSubs, nFolds,
            step);  // 0 for master id
  }
  delete[] td1->trialLengths;
  delete[] td2->trialLengths;
  delete[] td1->scs;
  delete[] td2->scs;
  //_mm_free(td1->data);
  //_mm_free(td2->data);
  free(td1->data);
  free(td2->data);
  delete td1;
  delete td2;
}

// function for sort
bool cmp(VoxelScore w1, VoxelScore w2) { return w1.score > w2.score; }

/* the master node splits the tasks and assign them to slave nodes.
When a slave nodes return, the master node would send another tasks to it.
The master node finally collects all results and sort them to write to a file.
The mask file here is the sample nifti file for writing */
void DoMaster(int nprocs, int step, int row, const char* output_file,
              const char* mask_file) {
#ifndef __MIC__
  using std::cout;
  using std::endl;
  using std::flush;
  using std::sort;
  using std::ofstream;

  int curSr = 0;
  int i, j;
  int total = row / step;
  if (row % step != 0) {
    total += 1;
  }
  cout << "total task: " << total << endl;

  int sentCount = 0;
  int doneCount = 0;
  int sendMsg[2];
  for (i = 1; i < nprocs; i++)  // fill up the processes first
  {
    sendMsg[0] = curSr;
    sendMsg[1] = step;
    if (curSr + step > row)  // overflow, so only get the left
    {
      sendMsg[1] = row - curSr;
    }
    curSr += sendMsg[1];
    MPI_Send(sendMsg,         /* message buffer, the correlation vector */
             2,               /* number of data to send */
             MPI_INT,         /* data item is float */
             i,               /* destination process rank */
             COMPUTATIONTAG,  /* user chosen message tag */
             MPI_COMM_WORLD); /* default communicator */
    sentCount++;
  }
  VoxelScore* scores = new VoxelScore[row];  // one voxel one score
  int totalLength = 0;
  MPI_Status status;
  while (sentCount < total) {
    int curLength;
    float elapse;
    // get the elapse time
    MPI_Recv(&elapse,        /* message buffer */
             1,              /* numbers of data to receive */
             MPI_FLOAT,      /* of type float real */
             MPI_ANY_SOURCE, /* receive from any sender */
             ELAPSETAG,      /* user chosen message tag */
             MPI_COMM_WORLD, /* default communicator */
             &status);       /* info about the received message */
    // get the length of message first
    MPI_Recv(&curLength,        /* message buffer */
             1,                 /* numbers of data to receive */
             MPI_INT,           /* of type float real */
             status.MPI_SOURCE, /* receive from any sender */
             LENGTHTAG,         /* user chosen message tag */
             MPI_COMM_WORLD,    /* default communicator */
             &status);          /* info about the received message */
    // get the classifier array
    MPI_Recv(scores + totalLength, /* message buffer */
             curLength * 2,        /* numbers of data to receive */
             MPI_FLOAT,            /* of type float real */
             status.MPI_SOURCE,    /* receive from the previous sender */
             VOXELCLASSIFIERTAG,   /* user chosen message tag */
             MPI_COMM_WORLD,       /* default communicator */
             &status);             /* info about the received message */
    totalLength += curLength;
    doneCount++;
    cout.precision(4);
    cout << doneCount << '\t' << elapse << "s\t" << flush;
    if (doneCount % 6 == 0) {
      cout << endl;
    }
    sendMsg[0] = curSr;
    sendMsg[1] = step;
    if (curSr + step > row)  // overflow, so only get the left
    {
      sendMsg[1] = row - curSr;
    }
    curSr += sendMsg[1];
    MPI_Send(sendMsg,           /* message buffer, the correlation vector */
             2,                 /* number of data to send */
             MPI_INT,           /* data item is float */
             status.MPI_SOURCE, /* destination process rank */
             COMPUTATIONTAG,    /* user chosen message tag */
             MPI_COMM_WORLD);   /* default communicator */
    sentCount++;
  }
  while (doneCount < total) {
    int curLength;
    float elapse;
    // get the elapse time
    MPI_Recv(&elapse,        /* message buffer */
             1,              /* numbers of data to receive */
             MPI_FLOAT,      /* of type float real */
             MPI_ANY_SOURCE, /* receive from any sender */
             ELAPSETAG,      /* user chosen message tag */
             MPI_COMM_WORLD, /* default communicator */
             &status);       /* info about the received message */
    // get the length of message first
    MPI_Recv(&curLength,        /* message buffer */
             1,                 /* numbers of data to receive */
             MPI_INT,           /* of type float real */
             status.MPI_SOURCE, /* receive from any sender */
             LENGTHTAG,         /* user chosen message tag */
             MPI_COMM_WORLD,    /* default communicator */
             &status);          /* info about the received message */
    // get the classifier array
    MPI_Recv(scores + totalLength, /* message buffer */
             curLength * 2,        /* numbers of data to receive */
             MPI_FLOAT,            /* of type float real */
             status.MPI_SOURCE,    /* receive from any sender */
             VOXELCLASSIFIERTAG,   /* user chosen message tag */
             MPI_COMM_WORLD,       /* default communicator */
             &status);             /* info about the received message */
    totalLength += curLength;
    doneCount++;
    cout.precision(4);
    cout << doneCount << '\t' << elapse << "s\t" << flush;
    if (doneCount % 6 == 0) {
      cout << endl;
    }
  }
  for (i = 1; i < nprocs; i++)  // tell all processes to stop
  {
    sendMsg[0] = -1;
    sendMsg[1] = -1;
    curSr += step;
    MPI_Send(sendMsg,         /* message buffer, the correlation vector */
             2,               /* number of data to send */
             MPI_INT,         /* data item is float */
             i,               /* destination process rank */
             COMPUTATIONTAG,  /* user chosen message tag */
             MPI_COMM_WORLD); /* default communicator */
  }
  sort(scores, scores + totalLength, cmp);
  cout << "Total length: " << totalLength << endl;
  // deal with output_file here////////////////////////////////////////////
  char fullfilename[MAXFILENAMELENGTH];
  sprintf(fullfilename, "%s", output_file);
  strcat(fullfilename, "_list.txt");
  ofstream ofile(fullfilename);
  for (j = 0; j < totalLength; j++) {
    ofile << scores[j].vid << " " << scores[j].score << endl;
  }
  ofile.close();
  if (mask_file) {
    int* data_ids = (int*)GenerateNiiDataFromMask(mask_file, scores, totalLength,
                                                DT_SIGNED_INT);
    sprintf(fullfilename, "%s", output_file);
    strcat(fullfilename, "_seq.nii.gz");
    WriteNiiGzData(fullfilename, mask_file, (void*)data_ids, DT_SIGNED_INT);
    float* data_scores = (float*)GenerateNiiDataFromMask(mask_file, scores,
                                                       totalLength, DT_FLOAT32);
    sprintf(fullfilename, "%s", output_file);
    strcat(fullfilename, "_score.nii.gz");
    WriteNiiGzData(fullfilename, mask_file, (void*)data_scores, DT_FLOAT32);
  }
  else {
    cout<<"the first mask is NULL, so no files in NIfTI format are generated"<<endl;
  }
#endif
}

/* the slave node listens to the master node and does the task that the master
node assigns.
A task is a matrix multiplication to get the correlation vectors of some voxels.
The slave node also does some preprocessing on the correlation vectors then
analyzes the correlatrion vectors (either do classification or compute the
average correlation coefficients.*/
void DoSlave(int me, int masterId, TrialData* td1, TrialData* td2,
             Task taskType, Trial* trials, int nTrials, int nHolds, int nSubs,
             int nFolds, int preset_step) {
  using std::cout;
  using std::endl;
  using std::flush;
  int recvMsg[2];
  MPI_Status status;
  size_t nVoxels = td1->nVoxels;
  size_t nVoxels2 = td2->nVoxels;
  Voxel* voxels = new Voxel();
  voxels->nTrials = nTrials;
  voxels->nVoxels = nVoxels;
  voxels->vid = new int[preset_step];
  //voxels->kernel_matrices = (float*)_mm_malloc(
  //    sizeof(float) * (size_t)nTrials * (size_t)nTrials * (size_t)preset_step,
  //    64);
  size_t dataSize = sizeof(float) * (size_t)nTrials * (size_t)nTrials * (size_t)preset_step;
  voxels->kernel_matrices = (float*)malloc(dataSize);
  assert(voxels->kernel_matrices);

  dataSize =
      sizeof(float) * (size_t)nVoxels2 * (size_t)BLK2 * (size_t)nTrials;
  if (1 == me) {
    cout << "task 1: bytes for correlation vecs: " << dataSize << endl << flush;
    if (getenv("FCMA_DEBUG_TASK")) WaitForDebugAttach();
  }

  //voxels->corr_vecs = (float*)_mm_malloc(dataSize, 64);
  voxels->corr_vecs = (float*)malloc(dataSize);
  assert(voxels->corr_vecs);
  while (true) {
    MPI_Recv(recvMsg,        /* message buffer */
             2,              /* numbers of data to receive */
             MPI_INT,        /* of type float real */
             masterId,       /* receive from any sender */
             COMPUTATIONTAG, /* user chosen message tag */
             MPI_COMM_WORLD, /* default communicator */
             &status);       /* info about the received message */
    double tstart = MPI_Wtime();
    int sr = recvMsg[0];
    int step = recvMsg[1];
    if (sr == -1)  // finish flag
    {
      break;
    }
    VoxelScore* scores = NULL;
    if (taskType == Corr_Based_SVM) {
#if __MEASURE_TIME__
      double t1 = MPI_Wtime();
#endif
      voxels =
          ComputeAllVoxelsAnalysisData(voxels, trials, nTrials, nSubs,
                                         nTrials - nHolds, sr, step, td1, td2);
// PreprocessAllVoxelsAnalysisData_flat(voxels, step, nSubs);
#if __MEASURE_TIME__
      double t2 = MPI_Wtime();
      cout << "computing: " << t2 - t1 << "s" << endl << flush;
#endif
      //scores = GetVoxelwiseSVMPerformance(
      //   me, trials, voxels, step, nTrials - nHolds, nFolds);  // LibSVM
      scores = GetVoxelwiseNewSVMPerformance(
            me, trials, voxels, step, nTrials - nHolds, nFolds); // PhiSVM
      // scores = new VoxelScore[step];
#if __MEASURE_TIME__
      t1 = MPI_Wtime();
      cout << "svm processing: " << t1 - t2 << "s" << endl;
#endif
    }
    else if (taskType == Corr_Based_Dis) {
      // TODO
      // it was distance ratio before
    }
    else if (taskType == Corr_Sum) {
      // scores = GetCorrVecSum(me, c_matrices, nTrials);
      // voxels = ComputeAllVoxelsAnalysisData(voxels, trials, nTrials, nSubs,
      // nTrials-nHolds, sr, step, td1, td2);
      scores = GetVoxelwiseCorrVecSum(me, voxels, step, sr, td1, td2);
    }
    else {
      FATAL("unknown task type");
    }
    double tstop = MPI_Wtime();
    float elapse = float(tstop - tstart);
    MPI_Send(&elapse,            /* message buffer, the correlation vector */
             1,                  /* number of data to send */
             MPI_FLOAT,          /* data item is float */
             masterId,           /* destination process rank */
             ELAPSETAG,          /* user chosen message tag */
             MPI_COMM_WORLD);    /* default communicator */
    MPI_Send(&step,              /* message buffer, the correlation vector */
             1,                  /* number of data to send */
             MPI_INT,            /* data item is float */
             masterId,           /* destination process rank */
             LENGTHTAG,          /* user chosen message tag */
             MPI_COMM_WORLD);    /* default communicator */
    MPI_Send(scores,             /* message buffer, the correlation vector */
             step * 2,           /* number of data to send */
             MPI_FLOAT,          /* data item is float */
             masterId,           /* destination process rank */
             VOXELCLASSIFIERTAG, /* user chosen message tag */
             MPI_COMM_WORLD);    /* default communicator */
    if (scores) {
      delete [] scores;
      scores = NULL;
    }
  }
  //_mm_free(voxels->corr_vecs);
  //_mm_free(voxels->kernel_matrices);
  free(voxels->corr_vecs);
  free(voxels->kernel_matrices);
  delete[] voxels->vid;
  delete voxels;
}
