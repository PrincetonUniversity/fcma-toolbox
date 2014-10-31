/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
 */

#ifndef FCMA_H
#define FCMA_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct param_t
{
    const char* fmri_directory;
    const char* fmri_file_type;
    const char* block_information_file;
    const char* block_information_directory;
    const char* mask_file1;
    const char* mask_file2;
    const char* ref_file;
    int step;
    const char* output_file;
    int leave_out_id;
    int taskType;
    int nHolds;
    int nFolds;
    int visualized_block_id;
    int isTestMode;
    int isUsingMaskFile;
    int isQuietMode;
    int shuffle;
    const char* permute_book_file;
}Param;

void run_fcma(Param* param);

#ifdef __cplusplus
}
#endif
#endif
