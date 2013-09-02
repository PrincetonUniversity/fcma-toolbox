/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include <mpi.h>
#include "common.h"
#include "Preprocessing.h"
#include "FileProcessing.h"
#include "Scheduler.h"
#include "SVMPredictor.h"
#include "SVMPredictorWithMasks.h"
#include "Searchlight.h"
#include "CorrelationVisualization.h"
#include "svm.h"

#include "fcma.h"

Param Parameters;
//extern unsigned long long counter;

void exit_with_help()
{
  printf(
  "**********************************************************\n"
  "Usage: [OMP_NUM_THREADS=n mpirun -np n -hostfile host] ./corr-sum -d matrix_directory -m matrix_file_type -t output_file -k taskType [options]\n"
  "required:\n"
  "-d matrix directory, contains files in binary gz format, beginning with numbers of row and column\n"
  "-m matrix file type, usually the extension name\n"
  "-k task type, 0 for voxel selection using svm, 1 for smart distance ratio, 2 for searchlight, 3 for correlation sum, 4 for two parts correlation and test, 5 for cross validation of two parts correlation, 6 for one part activation and test, 7 for cross validation of one part activation, 8 for voxel correlation visualizarion\n"
  "-t output file for task 0,1,2,3 in the voxel selection mode, input file for the same tasks in the test mode\n"
  "-b block information file, if no block information file, a block information directory is required\n"
  "-e block directory name, will check this if -b is not provided\n"
  "optional:\n"
  "-s step, the number of rows to be assigned per round, default 100\n"
  "-l leave out id, the first block id that being left out, default -1, which means don't leave out anything\n"
  "-h number of items that held for test, default -1, only and must get values when -l is applied\n"
  "-c bool, test mode (1) or not (0), default 0\n"
  "-n number of folds in the feature selection, default 0\n"
  "-x the first mask file, default no mask\n"
  "-y the second mask file, default no mask\n"
  "-v the block id that you want to visualize the correlation, must be specified in task 8\n"
  "-r the referred file of the output file of task 8, must be a 4D file, usually is the input data file\n"
  );
  exit(1);
}

void set_default_parameters()
{
  Parameters.fmri_directory = Parameters.fmri_file_type = Parameters.output_file = NULL;
  Parameters.block_information_file = Parameters.block_information_directory = NULL;
  Parameters.step = 100;
  Parameters.taskType = -1;
  Parameters.leave_out_id = -1;
  Parameters.nHolds = 0;
  Parameters.nFolds = -1;
  Parameters.visualized_block_id = -1;
  Parameters.isTestMode = false;
  Parameters.mask_file1=Parameters.mask_file2=NULL;
  Parameters.ref_file=NULL;
}

void check_parameters()
{
  if (Parameters.fmri_directory==NULL)
  {
    cout<<"no fmri directory, general information below"<<endl;
    exit_with_help();
  }
  if (Parameters.fmri_file_type==NULL)
  {
    cout<<"no fmri file type, general information below"<<endl;
    exit_with_help();
  }
  if (Parameters.block_information_file==NULL && Parameters.block_information_directory==NULL)
  {
    cout<<"no block information, general information below"<<endl;
    exit_with_help();
  }
  if (Parameters.taskType==-1)
  {
    cout<<"task type must be specified"<<endl;
    exit_with_help();
  }
  if (Parameters.output_file==NULL && (Parameters.taskType==0 || Parameters.taskType==1 || Parameters.taskType==2 || Parameters.taskType==3))
  {
    cout<<"no output file, general information below"<<endl;
    exit_with_help();
  }
  if (Parameters.leave_out_id>=0 && Parameters.nHolds==-1)
  {
    cout<<"number of holding samples must be specified"<<endl;
    exit_with_help();
  }
  if (!Parameters.isTestMode && Parameters.nFolds==-1)
  {
    cout<<"number of folds in the voxel selection must be specified"<<endl;
    exit_with_help();
  }
  if (Parameters.taskType==8 && Parameters.visualized_block_id==-1)
  {
    cout<<"the block to be visualized must be specified"<<endl;
    exit_with_help();
  }
  if (Parameters.taskType==8 && Parameters.ref_file==NULL)
  {
    cout<<"in the voxel correlation visualization task, a reference file must be provided"<<endl;
    exit_with_help();
  }
}

void parse_command_line(int argc, char **argv)
{
  set_default_parameters();
  int i;
  for (i=1; i<argc; i++)
  {
    if (argv[i][0] != '-')
    {
      break;
    }
    if (++i >= argc)
    {
      exit_with_help();
    }
    switch (argv[i-1][1])
    {
      case 's':
        Parameters.step = atoi(argv[i]);
        break;
      case 'd':
        Parameters.fmri_directory = argv[i];
        break;
      case 'm':
        Parameters.fmri_file_type = argv[i];
        break;
      case 'b':
        Parameters.block_information_file = argv[i];
        break;
      case 'e':
        Parameters.block_information_directory = argv[i];
        break;
      case 't':
        Parameters.output_file = argv[i];
        break;
      case 'k':
        Parameters.taskType = atoi(argv[i]);
        break;
      case 'l':
        Parameters.leave_out_id = atoi(argv[i]);
        break;
      case 'c':
        Parameters.isTestMode = (atoi(argv[i]) == 1);
        break;
      case 'h':
        Parameters.nHolds = atoi(argv[i]);
        break;
      case 'n':
        Parameters.nFolds = atoi(argv[i]);
        break;
      case 'x':
        Parameters.mask_file1 = argv[i];
        break;
      case 'y':
        Parameters.mask_file2 = argv[i];
        break;
      case 'v':
        Parameters.visualized_block_id = atoi(argv[i]);
        break;
      case 'r':
        Parameters.ref_file = argv[i];
        break;
      default:
        cout<<"unknown option: -"<<argv[i-1][1]<<endl;
        exit_with_help();
    }
  }
  check_parameters();
}

void run(Param* param)
{
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) MPI_Init(NULL,NULL);
    int me, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    /* initialization done */
    /* ---------------------------------------------- */
    /* set a series of input arguments */
    int step = param->step;
    int taskType = param->taskType;
    const char* fmri_directory = param->fmri_directory;
    const char* fmri_file_type = param->fmri_file_type;
    const char* block_information_file = param->block_information_file;
    const char* block_information_directory = param->block_information_directory;
    const char* output_file = param->output_file;
    const char* mask_file1 = param->mask_file1;
    const char* mask_file2 = param->mask_file2;
    const char* ref_file = param->ref_file;
    int leave_out_id = param->leave_out_id;
    int nHolds = param->nHolds; // the number of trials that being held from the analysis
    int nFolds = param->nFolds;
    int visualized_block_id = param->visualized_block_id;
    /* setting done */
    /* ---------------------------------------------- */
    /* data reading and initialization */
    int nTrials = 0;
    int nSubs = 0;
    RawMatrix** r_matrices = ReadGzDirectory(fmri_directory, fmri_file_type, nSubs);  // set nSubs here
    //Point* temp_pts=new Point[r_matrices[0]->row];
    //int row_tmp = AlignMatrices(r_matrices, nSubs, temp_pts);
    MPI_Barrier(MPI_COMM_WORLD); // wait for all nodes to finish reading the data
    if (me == 0)
    {
        cout<<"data reading done!"<<endl;
        //cout<<row_tmp<<endl;
    }
    Point* pts = ReadLocInfoFromNii(r_matrices[0]);  // assume that all subjects have the same format, so we can randomly pick one
    Trial* trials=NULL;
    if (block_information_file!=NULL)
    {
        trials = GenRegularTrials(nSubs, 0, nTrials, block_information_file);  // 0 for no shift, nTrials is assigned a value here
    }
    else
    {
        trials = GenBlocksFromDir(nSubs, 0, nTrials, r_matrices, block_information_directory);  // 0 for no shift, nTrials is assigned a value here
    }
    int nBlocksPerSub = nTrials / nSubs;  // assume each subject has the same number of blocks
    if (me == 0)
    {
        cout<<"blocks generation done! "<<nTrials<<" in total."<<endl;
        cout<<"blocks in the training set: "<<nTrials-nHolds<<endl;
    }
    MPI_Barrier(MPI_COMM_WORLD); // wait for all nodes to finish reading the data
    /* data reading done */
    /* ----------------------------------------------- */
    /* main program begins */
    double tstart = MPI_Wtime();
    //int row = r_matrices[0]->row;
    RawMatrix** avg_matrices = NULL;
    if (taskType == 2 || taskType == 6 || taskType == 7)
    {
        avg_matrices = rawMatPreprocessing(r_matrices, nSubs, nBlocksPerSub, trials); // 18 subjects, 12 blocks per subject
    }
    if (taskType != 5 && taskType != 7 && leave_out_id>=0)  // k==5 or 7 implies a cross validation
    {
        leaveSomeTrialsOut(trials, nTrials, leave_out_id, nHolds); // leave nHolds blocks out every time, usually 1 entire subject
    }
    if (param->isTestMode && me == 0) // program in test mode only uses one node to predict
    {
        int l = 0, result = 0, f = 0;
        cout<<"data directory: "<<fmri_directory<<endl;
        if (mask_file1!=NULL)
        {
            cout<<"mask file(s): "<<mask_file1<<" ";
            if (mask_file2!=NULL)
                cout<<mask_file2;
            cout<<endl;
        }
        cout<<"task type: "<<taskType<<endl;
        switch (taskType)
        {
            case 0:
            case 1:
            case 2:
            case 3:
                SVMPredict(r_matrices, avg_matrices, nSubs, nTrials, trials, nHolds, taskType, output_file, mask_file1);
                break;
            case 4:
                result = SVMPredictCorrelationWithMasks(r_matrices, nSubs, mask_file1, mask_file2, nTrials, trials, nHolds);
                cout<<"accuracy: "<<result<<"/"<<nHolds<<"="<<result*1.0/nHolds<<endl;
                break;
            case 5:
                nHolds = param->nHolds;
                while (l<=nTrials-nHolds) //assume that nHolds*an integer==nTrials
                {
                    leaveSomeTrialsOut(trials, nTrials, 0, nHolds); // the setting of third parameter is tricky here
                    int curResult = SVMPredictCorrelationWithMasks(r_matrices, nSubs, mask_file1, mask_file2, nTrials, trials, nHolds);
                    f++;
                    cout<<"fold "<<f<<": "<<curResult<<"/"<<nHolds<<"="<<curResult*1.0/nHolds<<endl;
                    result += curResult;
                    l += nHolds;
                }
                cout<<"total accuracy: "<<result<<"/"<<nTrials<<"="<<result*1.0/nTrials<<endl;
                break;
            case 6:
                result = SVMPredictActivationWithMasks(avg_matrices, nSubs, mask_file1, nTrials, trials, nHolds);
                cout<<"accuracy: "<<result<<"/"<<nHolds<<"="<<result*1.0/nHolds<<endl;
                break;
            case 7:
                nHolds = param->nHolds;
                while (l<=nTrials-nHolds) //assume that nHolds*an integer==nTrials
                {
                    leaveSomeTrialsOut(trials, nTrials, 0, nHolds); // the setting of third parameter is tricky here
                    int curResult = SVMPredictActivationWithMasks(avg_matrices, nSubs, mask_file1, nTrials, trials, nHolds);
                    f++;
                    cout<<"fold "<<f<<": "<<curResult<<"/"<<nHolds<<"="<<curResult*1.0/nHolds<<endl;
                    result += curResult;
                    l += nHolds;
                }
                cout<<"total accuracy: "<<result<<"/"<<nTrials<<"="<<result*1.0/nTrials<<endl;
                break;
            case 8:
                if (visualized_block_id>=nTrials)
                {
                    cerr<<"Wrong visualized block id, you only provide "<<nTrials<<" blocks!"<<endl;
                    exit(1);
                }
                VisualizeCorrelationWithMasks(r_matrices[0], mask_file1, mask_file2, ref_file, trials[visualized_block_id], output_file);
                break;
            default:
                cerr<<"Unknown task type"<<endl;
                exit_with_help();
        }
    }
    else if (!param->isTestMode)  //  program not in test mode is in voxel selction mode, which will use multiple nodes if doing correlation based selection
    {
        // compute the matrix, do analysis
        switch (taskType)
        {
            case 0:
            case 1:
            case 3:
                if (me==0)  // master process doesn't need to keep r_matrices
                {
                    if (taskType==0) cout<<"SVM selecting..."<<endl;
                    if (taskType==1) cout<<"distance ratio selecting..."<<endl;
                    if (taskType==3) cout<<"correlation sum..."<<endl;
                }
                Scheduler(me, nprocs, step, r_matrices, taskType, trials, nTrials, nHolds, nSubs, nFolds, output_file, mask_file1, mask_file2);
                break;
            case 2:
                cout<<"Searchlight selecting..."<<endl;
                Searchlight(avg_matrices, nSubs, trials, nTrials, nHolds, nFolds, pts, output_file, mask_file1);  // doesn't need mpi
                break;
            default:
                cerr<<"Unknown task type"<<endl;
                exit_with_help();
        }
    }
    double tstop = MPI_Wtime();
    if (me == 0)
    {
        cout.precision(6);
        cout<<"it takes "<<tstop-tstart<<" s to complete the whole task (exclude data reading)"<<endl;
    }
    /* main program ends */
    /* -------------------------------- */
    /* Shut down MPI */
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) MPI_Finalize();

    //cout<<counter<<endl;
}

#ifndef NOMAIN

int main(int argc, char** argv)
{
    //counter=0;
    parse_command_line(argc, argv);
    run_fcma(&Parameters);
    return 0;
}

#endif

