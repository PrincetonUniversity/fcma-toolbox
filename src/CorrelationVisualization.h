/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"

void VisualizeCorrelationWithMasks(RawMatrix* r_matrix, const char* maskFile1,
                                   const char* maskFile2, const char* refFile,
                                   Trial trial, const char* output_file);
float* PutMaskedDataBack(const char* maskFile, float* data, int row, int col);
