/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include <cstring>
#include "common.h"

RawMatrix** ReadGzDirectory(const char* filepath, const char* filetype,
                            int& nSubs);
RawMatrix* ReadGzData(std::string fileStr, int sid);
RawMatrix* ReadNiiGzData(std::string fileStr, int sid);
RawMatrix** GetMaskedMatrices(RawMatrix** r_matrices, int nSubs,
                              const char* maskFile, bool deleteData);
RawMatrix* GetMaskedMatrix(RawMatrix* r_matrix, const char* maskFile);
void GenerateMaskedMatrices(int nSubs, RawMatrix** r_matrices1, RawMatrix** r_matrices2,
                            const char* mask_file1, const char* mask_file2,
                            RawMatrix*** p_masked_matrices1, RawMatrix*** p_masked_matrices2);
VoxelXYZ* GetMaskedPts(VoxelXYZ* pts, int nMaskedVoxels, const char* maskFile);
Trial* GenRegularTrials(int nSubs, int nShift, int& nTrials, const char* file);
Trial* GenBlocksFromDir(int nSubs, int nShift, int& nTrials,
                        RawMatrix** r_matrices, const char* dir);
VoxelXYZ* ReadLocInfo(const char* file);
VoxelXYZ* ReadLocInfoFromNii(RawMatrix* r_matrix);
double** ReadRTMatrices(const char* file, int& nSubs);
void WriteNiiGzData(const char* outputFile, const char* refFile, void* data,
                    int dataType);
void Write4DNiiGzData(const char* outputFile, const char* refFile, void* data,
                      int dataType, int nt);
void* GenerateNiiDataFromMask(const char* maskFile, VoxelScore* scores,
                              int length, int dataType);
inline int getSizeByDataType(int datatype);
const char* GetFilenameExtension(const char* filename);
int ReadConfigFile(const char* fcma_file, const int& length,
                   char** keys_and_values);

void WriteCorrMatToHDF5(int row1, int row2, float* corrMat, const char* outputfile);
