#include <cstring>
#include "common.h"
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
extern "C" {
#include <cblas.h>
}
#endif

RawMatrix** ReadGzDirectory(const char* filepath, const char* filetype, int& nSubs);
RawMatrix* ReadGzData(string fileStr, int sid);
RawMatrix* ReadNiiGzData(string fileStr, int sid);
RawMatrix** GetMaskedMatrices(RawMatrix** r_matrices, int nSubs, const char* maskFile);
RawMatrix* GetMaskedMatrix(RawMatrix* r_matrix, const char* maskFile);
Point* GetMaskedPts(Point* pts, int nMaskedVoxels, const char* maskFile);
Trial* GenRegularTrials(int nSubs, int nShift, int& nTrials, const char* file);
Trial* GenBlocksFromDir(int nSubs, int nShift, int& nTrials, RawMatrix** r_matrices, const char* dir);
Point* ReadLocInfo(const char* file);
Point* ReadLocInfoFromNii(RawMatrix* r_matrix);
double** ReadRTMatrices(const char* file, int& nSubs);
void WriteNiiGzData(const char* outputFile, const char* refFile, void* data, int dataType);
void Write4DNiiGzData(const char* outputFile, const char* refFile, void* data, int dataType, int nt);
void* GenerateNiiDataFromMask(const char* maskFile, VoxelScore* scores, int length, int dataType);
inline int getSizeByDataType(int datatype);
