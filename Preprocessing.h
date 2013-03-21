#include "common.h"
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
extern "C" {
#include <cblas.h>
}
#endif

RawMatrix** ReadGzDirectory(const char* filepath, const char* filetype, int& nSubs);
RawMatrix* ReadGzData(const char* file, int sid);
RawMatrix* ReadNiiGzData(const char* file, int sid);
int AlignMatrices(RawMatrix** r_matrices, int nSubs, Point* pts);
int AlignMatricesByFile(RawMatrix** r_matrices, int nSubs, const char* file, Point* pts);
RawMatrix** GetMaskedMatrices(RawMatrix** r_matrices, int nSubs, const char* maskFile);
Trial* GenRegularTrials(int nSubs, int nShift, int& nTrials, const char* file);
void leaveSomeTrialsOut(Trial* trials, int nTrials, int tid, int nLeaveOut);
void corrMatPreprocessing(CorrMatrix** c_matrices, int n, int nSubs);
float fisherTransformation(float v);
void z_score(double* v, int n);
Point* ReadLocInfo(const char* file);
Point* ReadLocInfoFromNii(RawMatrix* r_matrix);
double** ReadRTMatrices(const char* file, int& nSubs);
RawMatrix** rawMatPreprocessing(RawMatrix** r_matrices, int n, int nTrials, Trial* trials);
float getAverage(RawMatrix* r_matrix, Trial trial, int vid);
