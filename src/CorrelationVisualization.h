#include "common.h"

void VisualizeCorrelationWithMasks(RawMatrix* r_matrix, const char* maskFile1, const char* maskFile2, Trial trial, const char* output_file);
float* PutMaskedDataBack(const char* maskFile, float* data, int row, int col);
