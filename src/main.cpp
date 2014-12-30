/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include <signal.h>
#include <mpi.h>
#include "common.h"
#include "Preprocessing.h"
#include "FileProcessing.h"
#include "Scheduler.h"
#include "Marginal_Scheduler.h"
#include "SVMPredictor.h"
#include "SVMPredictorWithMasks.h"
#include "Searchlight.h"
#include "CorrelationVisualization.h"
#include "svm.h"
#include "fcma.h"
#include "ErrorHandling.h"

Param Parameters;
//extern unsigned long long counter;

static void params_from_keyvalues(char** keys_and_values,const int& num_elements);

void exit_with_help()
{
  printf(
  "**********************************************************\n"
  "Usage 1: [OMP_NUM_THREADS=n mpirun -np n -hostfile host] ./pni_fcma -d matrix_directory -m matrix_file_type -t output_file -k taskType [options]\n"
  "Usage 2: [OMP_NUM_THREADS=n mpirun -np n -hostfile host] ./pni_fcma config.fcma\n"
  "required:\n"
  "-d nifti file directory, contains files in nifti gz format\n"
  "-1 one of the nifti file directories, if providing -1, -2 must be provided, too. providing -1 and -2 will cause error if providing -d as well\n"
  "-2 the other one of the nifti file directories, -1 and -2 MUST have the same number of nifti files and file name sequence should be matched. -2 won't be used in activity-based analysis\n"
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
  "-y the second mask file, default no mask. for voxel selection, we generally require mask1==mask2\n"
  "-v the block id that you want to visualize the correlation, must be specified in task 8\n"
  "-r the referred file of the output file of task 8, must be a 4D file, usually is the input data file\n"
  "-q bool, being quiet (1) or not (0) in test mode for task type 2 4 5 6 7, default 0\n"
  "-f randomly shuffle the voxel data, 0-no shuffling, 1-shuffle using a random time seed (results are not repeatable), 2-shuffle using an input permuting book (results are repeatable), default 0\n"
  "-p permuting book, only get values when -f is set to 2; the book contains #subjects row and #voxels column; each row contains random shuffling result of 0 to #voxels-1\n"
  );
  FATAL("usage");
}

void set_default_parameters()
{
  Parameters.fmri_directory = NULL;
  Parameters.fmri_directory1 = NULL;
  Parameters.fmri_directory2 = NULL;
  Parameters.fmri_file_type = ".nii.gz";
  Parameters.output_file = "topvoxels";
  Parameters.block_information_file = Parameters.block_information_directory = NULL;
  Parameters.step = 100;
  Parameters.taskType = 0;
  Parameters.leave_out_id = -1;
  Parameters.nHolds = 0;
  Parameters.nFolds = -1;
  Parameters.visualized_block_id = -1;
  Parameters.isTestMode = false;
  Parameters.mask_file1=Parameters.mask_file2=NULL;
  Parameters.ref_file=NULL;
  Parameters.isQuietMode = false;
  Parameters.shuffle=0;
  Parameters.permute_book_file=NULL;
}

void check_parameters()
{
  if (Parameters.fmri_directory==NULL && Parameters.fmri_directory1==NULL && Parameters.fmri_directory2==NULL)
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
  if (Parameters.shuffle==2 && Parameters.permute_book_file==NULL)
  {
    cout<<"the permute book file should be specified"<<endl;
    exit_with_help();
  }
  if (Parameters.shuffle!=2 && Parameters.permute_book_file!=NULL)
  {
    cout<<"the permute book file should not be set if -f is not 2"<<endl;
    exit_with_help();
  }
  if (Parameters.fmri_directory!=NULL && (Parameters.fmri_directory1!=NULL || Parameters.fmri_directory2!=NULL))
  {
    cout<<"-d and -1 -2 cannot be used together"<<endl;
    exit_with_help();
  }
  if (Parameters.fmri_directory==NULL && (Parameters.fmri_directory1==NULL || Parameters.fmri_directory2==NULL))
  {
    cout<<"-1 -2 should be provided together"<<endl;
    exit_with_help();
  }
}

void parse_command_line(int argc, char **argv)
{
#ifndef __MIC__
  set_default_parameters();

  if (argc < 2) {
    exit_with_help();
  }

  int start = 1;
  if (argv[1][0] != '-')
  {
    const char* ext = GetFilenameExtension(argv[1]);
    if (!strncasecmp(ext,"fcma",4))
    {
        //std::cout << "reading fcma config file" << std::endl;
        int max_elements = 256;
        char** keys_and_values = new char*[max_elements];
        int num_elements = ReadConfigFile(argv[1], max_elements, keys_and_values);
        params_from_keyvalues(keys_and_values,num_elements);
        start = 2;
    }
  }

  for (int i=start+1; i<argc; i+=2)  // here i should increase by two, but the code here is not safe now
  {
    switch (argv[i-1][1])
    {
      case 's':
        Parameters.step = atoi(argv[i]);
        break;
      case 'd':
        Parameters.fmri_directory = argv[i];
        break;
      case '1':
        Parameters.fmri_directory1 = argv[i];
        break;
      case '2':
        Parameters.fmri_directory2 = argv[i];
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
      case 'q':
        Parameters.isQuietMode = (atoi(argv[i]) == 1);
        break;
      case 'f':
        Parameters.shuffle = atoi(argv[i]);
        break;
      case 'p':
        Parameters.permute_book_file = argv[i];
        break;
      default:
        cout<<"unknown option: -"<<argv[i-1][1]<<endl;
        exit_with_help();
    }
  }
  check_parameters();
#endif
}

#define KEYS_DEF \
KEY_DEF( FCMA_DATADIR, "datadir" ),  \
KEY_DEF( FCMA_MATRIX_FORMAT,  "matrix_format" ),   \
KEY_DEF( FCMA_OUTPUTFILE, "outputfile" ), \
KEY_DEF( FCMA_TASK_TYPE, "task_type" ), \
KEY_DEF( FCMA_BLOCKFILE, "blockfile") , \
KEY_DEF( FCMA_BLOCKDIR, "blockdir" ), \
KEY_DEF( FCMA_ROWS_PER_ROUND, "rows_per_round" ), \
KEY_DEF( FCMA_FIRST_LEFT_OUT_BLOCK_ID, "first_left_out_block_id" ), \
KEY_DEF( FCMA_NUM_ITEMS_HELD_FOR_TEST, "num_items_held_for_test" ), \
KEY_DEF( FCMA_IS_TEST_MODE, "is_test_mode" ), \
KEY_DEF( FCMA_NUM_FOLDS_IN_FEATURE_SELECTION, "num_folds_in_feature_selection" ), \
KEY_DEF( FCMA_FIRST_MASKFILE, "first_maskfile" ), \
KEY_DEF( FCMA_SECOND_MASKFILE, "second_maskfile" ), \
KEY_DEF( FCMA_VISUALIZE_BLOCKID, "visualize_blockid" ), \
KEY_DEF( FCMA_VISUALIZE_REFERENCE, "visualize_reference" )

#define KEY_DEF( identifier, name )  identifier
enum keys { KEYS_DEF };

enum
{
    FCMA_FIRSTKEY = FCMA_DATADIR,
    FCMA_LASTKEY = FCMA_VISUALIZE_REFERENCE,
    FCMA_NUM_KEYS = FCMA_LASTKEY + 1
};

#undef KEY_DEF
#define KEY_DEF( identifier, name )  name
const char* values[] = { KEYS_DEF };

static void params_from_keyvalues(char** keys_and_values,const int& num_elements)
{
  bool found[FCMA_NUM_KEYS];
  for (int i=0; i<FCMA_NUM_KEYS; i++) found[i] = false;

  for ( int i=0; i<num_elements; i+=2 )
  {
    int v = i+1;
    for ( int k = FCMA_FIRSTKEY; k < FCMA_NUM_KEYS; k++ )
    {
      if ( found[k] ) continue;
      if ( !strcmp(keys_and_values[i],values[k]) )
      {
        switch( k )
        {
          case FCMA_DATADIR:
            Parameters.fmri_directory = keys_and_values[v];
            break;
          case FCMA_MATRIX_FORMAT:
            Parameters.fmri_file_type = keys_and_values[v];
            break;
          case FCMA_OUTPUTFILE:
            Parameters.output_file = keys_and_values[v];
            break;
          case FCMA_TASK_TYPE:
            Parameters.taskType = atoi(keys_and_values[v]);
            break;
          case FCMA_BLOCKFILE:
            Parameters.block_information_file = keys_and_values[v];
            break;
          case FCMA_BLOCKDIR:
            Parameters.block_information_directory = keys_and_values[v];
            break;
          case FCMA_ROWS_PER_ROUND:
            Parameters.step = atoi(keys_and_values[v]);
            break;
          case FCMA_FIRST_LEFT_OUT_BLOCK_ID:
            Parameters.leave_out_id = atoi(keys_and_values[v]);
            break;
          case FCMA_NUM_ITEMS_HELD_FOR_TEST:
            Parameters.nHolds = atoi(keys_and_values[v]);
            break;
          case FCMA_IS_TEST_MODE:
            Parameters.isTestMode = atoi(keys_and_values[v]);
            break;
          case FCMA_NUM_FOLDS_IN_FEATURE_SELECTION:
            Parameters.nFolds = atoi(keys_and_values[v]);
            break;
          case FCMA_FIRST_MASKFILE:
            Parameters.mask_file1 = keys_and_values[v];
            break;
          case FCMA_SECOND_MASKFILE:
            Parameters.mask_file2 = keys_and_values[v];
            break;
          case FCMA_VISUALIZE_BLOCKID:
            Parameters.visualized_block_id = atoi(keys_and_values[v]);
            break;
          case FCMA_VISUALIZE_REFERENCE:
            Parameters.ref_file = keys_and_values[v];
            break;
          default:
            break;
        }
        found[k] = true;
      }
    }
  }
}

void run_fcma(Param* param)
{
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) MPI_Init(NULL,NULL);
    int me, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    if (me==0)
    {
      cout<<"The program runs "<<nprocs<<"in total"<<endl;
    }
    /* initialization done */
    /* ---------------------------------------------- */
    /* set a series of input arguments */
    int step = param->step;
    int taskType = param->taskType;
    const char* fmri_directory = param->fmri_directory;
    if (fmri_directory==NULL) fmri_directory = param->fmri_directory1;
    const char* fmri_directory2 = param->fmri_directory2;
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
    int is_quiet_mode = param->isQuietMode;
    int shuffle = param->shuffle;
    const char* permute_book_file = param->permute_book_file;
    /* setting done */
    /* ---------------------------------------------- */
    /* data reading and initialization */
    // workers don't read in parameters, so some parameters need to be broadcasted
    MPI_Bcast((void*)&nHolds, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*)&leave_out_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*)&nFolds, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*)&taskType, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int nTrials = 0;
    int nSubs = 0;

    RawMatrix** r_matrices = NULL;
    RawMatrix** r_matrices2 = NULL;
#ifndef __MIC__
    if (me==0)
    {
      r_matrices = ReadGzDirectory(fmri_directory, fmri_file_type, nSubs);  // set nSubs here
      if (fmri_directory2)
      {
        r_matrices2 = ReadGzDirectory(fmri_directory2, fmri_file_type, nSubs);  // set nSubs here
      }
      else
      {
        r_matrices2 = r_matrices;
      }
    }
#endif
    MPI_Bcast((void*)&nSubs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //VoxelXYZ* temp_pts=new VoxelXYZ[r_matrices[0]->row];
    //int row_tmp = AlignMatrices(r_matrices, nSubs, temp_pts);
    MPI_Barrier(MPI_COMM_WORLD); // wait for all nodes to finish reading the data
    if (me == 0)
    {
        cout<<"data reading done!"<<endl;
        //cout<<row_tmp<<endl;
    }
    VoxelXYZ* pts = NULL; 
#ifndef __MIC__
    if (me == 0)
    {
      pts = ReadLocInfoFromNii(r_matrices[0]);  // assume that all subjects are aligned; use first subject's voxel coordinates
    }
#endif
    Trial* trials=NULL;
#ifndef __MIC__
    if (me == 0)
    {
      if (block_information_file!=NULL)
      {
        trials = GenRegularTrials(nSubs, 0, nTrials, block_information_file);  // 0 for no shift, nTrials is assigned a value here
      }
      else
      {
        trials = GenBlocksFromDir(nSubs, 0, nTrials, r_matrices, block_information_directory);  // 0 for no shift, nTrials is assigned a value here
      }
    }
#endif
    MPI_Bcast((void*)&nTrials, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (me != 0)
    {
      trials = new Trial[nTrials];
    }
    MPI_Bcast((void*)trials, sizeof(Trial)*nTrials, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (me == 0)
    {
        cout<<"blocks generation done! "<<nTrials<<" in total."<<endl;
        cout<<"blocks in the training set: "<<nTrials-nHolds<<endl;
        if (nHolds > nTrials)
            FATAL("More holds ("<<nHolds<<") than trials ("<<nTrials<<")!");
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
        int nBlocksPerSub = nTrials / nSubs;  // assume each subject has the same number of blocks
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
                SVMPredict(r_matrices, r_matrices2, avg_matrices, nSubs, nTrials, trials, nHolds, taskType, output_file, mask_file1, is_quiet_mode);
                break;
            case 4:
                result = SVMPredictCorrelationWithMasks(r_matrices, r_matrices2, nSubs, mask_file1, mask_file2, nTrials, trials, nHolds, is_quiet_mode);
                cout<<"accuracy: "<<result<<"/"<<nHolds<<"="<<result*1.0/nHolds<<endl;
                break;
            case 5:
                nHolds = param->nHolds;
                while (l<=nTrials-nHolds) //assume that nHolds*an integer==nTrials
                {
                    leaveSomeTrialsOut(trials, nTrials, 0, nHolds); // the setting of third parameter is tricky here
                    int curResult = SVMPredictCorrelationWithMasks(r_matrices, r_matrices2, nSubs, mask_file1, mask_file2, nTrials, trials, nHolds, is_quiet_mode);
                    f++;
                    cout<<"fold "<<f<<": "<<curResult<<"/"<<nHolds<<"="<<curResult*1.0/nHolds<<endl;
                    result += curResult;
                    l += nHolds;
                }
                cout<<"total accuracy: "<<result<<"/"<<nTrials<<"="<<result*1.0/nTrials<<endl;
                break;
            case 6:
                result = SVMPredictActivationWithMasks(avg_matrices, nSubs, mask_file1, nTrials, trials, nHolds, is_quiet_mode);
                cout<<"accuracy: "<<result<<"/"<<nHolds<<"="<<result*1.0/nHolds<<endl;
                break;
            case 7:
                nHolds = param->nHolds;
                while (l<=nTrials-nHolds) //assume that nHolds*an integer==nTrials
                {
                    leaveSomeTrialsOut(trials, nTrials, 0, nHolds); // the setting of third parameter is tricky here
                    int curResult = SVMPredictActivationWithMasks(avg_matrices, nSubs, mask_file1, nTrials, trials, nHolds, is_quiet_mode);
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
                    FATAL("Wrong visualized block id, you only provide "<<nTrials<<" blocks!");
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
                Scheduler(me, nprocs, step, r_matrices, r_matrices2, taskType, trials, nTrials, nHolds, nSubs, nFolds, output_file, mask_file1, mask_file2, shuffle, permute_book_file);
                break;
            case 2:
                cout<<"Searchlight selecting..."<<endl;
                Searchlight(avg_matrices, nSubs, trials, nTrials, nHolds, nFolds, pts, output_file, mask_file1, shuffle, permute_book_file);  // doesn't need mpi
                break;
            case 9:
                if (me==0)
                {
                  cout<<"marginal screening starting..."<<endl;
                }
                Marginal_Scheduler(me, nprocs, step, r_matrices, taskType, trials, nTrials, nHolds, nSubs, nFolds, output_file, mask_file1);
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
        cout<<"it takes "<<tstop-tstart<<" s to compute the whole task (exclude data reading but include data transferring using MPI)"<<endl;
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
    //raise(SIGSTOP); //xcode attach to process
    
    parse_command_line(argc, argv);
    run_fcma(&Parameters);
    return 0;
}

#endif

