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
