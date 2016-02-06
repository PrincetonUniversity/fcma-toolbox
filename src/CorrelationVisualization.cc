/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#ifndef __MIC__

#include "CorrelationVisualization.h"
#include "FileProcessing.h"
#include "CustomizedMatrixMultiply.h"
#include "ErrorHandling.h"

static void mat2buf(float* buf, float* mat, int voxels, int alltrs, int trs, int sc, bool accum) {
    long k=0;
    for (int i=0; i<voxels; ++i) {
        long voffset = alltrs * i + sc;
        for (int j=0; j<trs; ++j) {
            float val = mat[voffset + j];
            if (accum) {
                buf[k] += val;
            } else {
                buf[k] = val;
            }
            ++k;
        }
    }
}
                    

static float matmean(float* mat, long n) {
    double ac = 0.0f;
    for (long i=0; i<n; ++i) {
        ac += (double)mat[i];
    }
    ac /= (double)n;
    return (float)ac;
}

static void normbuf(float* buf, int voxels, int trs) {
    double trs_d = (double)trs;
    for (int i = 0; i < voxels; ++i) {
        double mean = 0.0f;
        double sd = 0.0f;
        long voffset = i * trs;
        
        for (int j = 0; j < trs; ++j) {
            double val = (double)buf[voffset + j];
            mean += val;
            sd += val * val;
        }
        mean /= trs_d;
    
        ALIGNED(64) float inv_sd_f;
        ALIGNED(64) float mean_f = mean;
        sd = sqrt(sd - trs_d * mean * mean);
        if (sd == 0.0f) {
            inv_sd_f = 0.0f;
        } else {
            inv_sd_f = 1.0f / sd;
        }
#pragma simd
        for (int j = 0; j < trs; ++j) {
            buf[voffset + j] -= mean_f;
            buf[voffset + j] *= inv_sd_f;
        }
    }
}

static void divmat(float* buf, long elements, float divisor) {
    for (long i=0; i<elements; ++i) {
        buf[i] /= divisor;
    }
}

/***************************************
Compute an average correlation matrix constructed this way:
   - correlate each subject with all other subjects
           ie subject 1 with subjects 2...N
              subject 2 with subjects 1, 3.. N
       etc
       by summing the values at each voxel across the other
       N - 1 subjects, for the TRs in the given trial, and correlating.
      (Where correlations are dot products between z-scored vectors)
   - Then take mean of the N correlation matrices 
 ****************************************/
void WriteAverageCorrelations(int nSubs, RawMatrix** r_matrices,
                              const char* maskFile1, const char* maskFile2,
                              Trial trial, const char* output_file) {
    using std::cout;
    using std::endl;
    using std::flush;
    
    // 1 get the dimensions of the problem
    RawMatrix* masked_matrix1 = NULL;
    RawMatrix* masked_matrix2 = NULL;
    if (maskFile1 != NULL)
        masked_matrix1 = GetMaskedMatrix(r_matrices[0], maskFile1);
    else
        masked_matrix1 = r_matrices[0];
    
    int voxels = masked_matrix1->row;
    int alltrs = masked_matrix1->col;
    int sc = trial.sc;
    int ec = trial.ec;
    assert(ec > sc);
    int trs = ec - sc + 1;

    cout << "#voxels in mask:" << voxels << endl;
    cout << "# TRs in trial: " << trs << endl;
    
    long correlations = (long)voxels * (long)voxels;
    long elements = (long)voxels * (long)trs;
    
    // 2 allocate the matrices
    float* buf1 = (float*)calloc(elements, sizeof(float));
    float* acc = (float*)calloc(elements, sizeof(float));
    // corrMat needs to be set to 0 since it will be accumulated
    // - the other ones are initialized prior to use.
    float* corrMat = (float*)calloc(correlations, sizeof(float));

    int completed_correlations = 0;

    // for all subjects ...
    for (int s1=0; s1<nSubs; ++s1) {
        
        // get the data for that subject into our trial-specific matrix "buf1"
        // first get the whole matrix, across all trials
        if (maskFile1 != NULL)
            masked_matrix1 = GetMaskedMatrix(r_matrices[s1], maskFile1);
        else
            masked_matrix1 = r_matrices[s1];
        
        // now pull out and normalize the data for the trial in question into buf1
        mat2buf(buf1, masked_matrix1->matrix, voxels, alltrs, trs, sc, false);
        normbuf(buf1, voxels, trs);
        
        memset((void*)acc, 0, elements*sizeof(float));
        
        for (int s2=0; s2<nSubs; ++s2) {
            
            // don't include same subject (the "1" in "N-1")
            if (s1 == s2) continue;
            
            // get the matrix across all trials for next other subject
            if (maskFile2 != NULL)
                masked_matrix2 = GetMaskedMatrix(r_matrices[s2], maskFile2);
            else
                masked_matrix2 = r_matrices[s2];
            
            // now pull out this trial's data and add it to "acc"
            mat2buf(acc, masked_matrix2->matrix, voxels, alltrs, trs, sc, true);
            
        } // loop accumulating N-1 brains
        
        // here we need to normalize the summed brain
        normbuf(acc, voxels, trs);
        
        // now do the "correlation" (dot products between z-scored vectors), for this trial, between
        // all voxel pairs, subject 1 ("buf1") vs all other subjects ("acc") and store in "corrMat"
        // which will accumulate across the N subjects (since beta on it is 1.0, weights input C[orrmat])
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, voxels, voxels, trs, 1.0,
                    buf1, trs, acc, trs, 1.0, corrMat, voxels);

        
        ++completed_correlations;

#ifdef DEBUG
        cout << "done correlating all other subjects with subject"  << s1 << endl << flush;
 //       cout << "[ global average correlation = " << matmean(corrMat, correlations, completed_correlations) << " ]" << endl << flush;
#endif
        
    } // loop acccumulating correlation matrix
    free(buf1);
    free(acc);
    
    divmat(corrMat, voxels*voxels, completed_correlations);
    cout << "[ global average correlation for this trial = " << matmean(corrMat, correlations) << " ]" << endl << flush;
   
    WriteCorrMatToHDF5(voxels, voxels, corrMat, output_file);
    
    free(corrMat);
}

#endif
