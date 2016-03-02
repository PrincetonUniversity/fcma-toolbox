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


struct fcma_timer {
    struct timeval start;
    std::string event_name;
    fcma_timer(std::string in_event_name) : event_name(in_event_name) {
        gettimeofday(&start, 0);
    }
    ~fcma_timer(void) {
        struct timeval end;
        gettimeofday(&end, 0);
        std::cerr << "time for " << event_name << " = " << (end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 0.000001) << std::endl;
    }
};

#ifdef __MEASURE_TIME__

static fcma_timer* fcma_start_timer(const std::string& event_name) {
    fcma_timer* tm = new fcma_timer(event_name);
    return tm;
}

static void fcma_stop_timer(fcma_timer* tm) {
    assert(tm);
    delete tm;
    tm = nil;
}

#   define START_TIMER(tm, event_name) fcma_timer* tm = fcma_start_timer(event_name)
#   define STOP_TIMER(tm) fcma_stop_timer(tm)
#else
#   define START_TIMER(start) return nil
#   define STOP_TIMER(end) ((void)0)
#endif

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
    using std::string;
 
    START_TIMER(total_timer, "correlation matrix total time");
    START_TIMER(setup_timer, "setup");
    
    // 1 get the dimensions of the problem
    
    // first get all the masked matrices
    START_TIMER(mask_timer, "get subject masked data, for both masks, from raw data");
    RawMatrix** masked_matrices1 = new RawMatrix*[nSubs];
    RawMatrix** masked_matrices2 = new RawMatrix*[nSubs];
    for (int i = 0; i<nSubs; ++i) {
        
        if (maskFile1 != NULL)
            masked_matrices1[i] = GetMaskedMatrix(r_matrices[i], maskFile1);
        else
            masked_matrices1[i] = r_matrices[i];
        
        if (maskFile2 != NULL)
            masked_matrices2[i] = GetMaskedMatrix(r_matrices[i], maskFile2);
        else
            masked_matrices2[i] = r_matrices[i];
        
        if (maskFile1 != NULL && maskFile2 != NULL) {
            delete [] r_matrices[i]->matrix;
            r_matrices[i]->matrix = NULL;
        }
    }
    STOP_TIMER(mask_timer);
    
    
    int voxels1 = masked_matrices1[0]->row;
    int voxels2 = masked_matrices2[0]->row;
    assert(masked_matrices1[0]->col == masked_matrices2[0]->col);
    int alltrs = masked_matrices1[0]->col;
    
    int sc = trial.sc;
    int ec = trial.ec;
    assert(ec > sc);
    int trs = ec - sc + 1;

    cout << "#voxels in mask1:" << voxels1 << endl;
    cout << "#voxels in mask2:" << voxels2 << endl;
    cout << "# TRs in trial: " << trs << endl;
    
    long correlations = (long)voxels1 * (long)voxels2;
    long elements1 = (long)voxels1 * (long)trs;
    long elements2 = (long)voxels2 * (long)trs;
    
    cout << "matrix size for this block in mask1 = " << elements1 << endl;
    cout << "matrix size for this block in mask2 = " << elements2 << endl;
    cout << "matrix size for correlations is voxels in mask1 * voxels in mask2 = " << correlations << endl;
    cout << "total allocating " << (float)((elements1 + elements2 + correlations)*sizeof(float))/(1024.0 * 1024.0 * 1024.0) << " GB for computing correlation matrix" << endl;
    
    // 2 allocate the matrices
    float* buf1 = (float*)calloc(elements1, sizeof(float)); // mask1
    float* acc = (float*)calloc(elements2, sizeof(float));  // mask2 being built up over subject
    // corrMat needs to be set to 0 since it will be accumulated
    // - the other ones are initialized prior to use.
    float* corrMat = (float*)calloc(correlations, sizeof(float));

    int completed_correlations = 0;

    STOP_TIMER(setup_timer);
    
    // for all subjects ...
    for (int s1=0; s1<nSubs; ++s1) {
        START_TIMER(subject_timer, "subject");
        
        // now pull out and normalize the data for the trial in question into buf1
        mat2buf(buf1, masked_matrices1[s1]->matrix, voxels1, alltrs, trs, sc, false);
        delete [] masked_matrices1[s1]->matrix; masked_matrices1[s1]->matrix = NULL;
        delete masked_matrices1[s1];
        normbuf(buf1, voxels1, trs);
        
        // build N-1 brain
        START_TIMER(sumbrain_timer, "N-1 brain calculation");
        memset((void*)acc, 0, elements2*sizeof(float));
        for (int s2=0; s2<nSubs; ++s2) {
            
            // don't include same subject (the "1" in "N-1")
            if (s1 == s2) continue;
            
            // now pull out this trial's data and add it to "acc"
            mat2buf(acc, masked_matrices2[s2]->matrix, voxels2, alltrs, trs, sc, true);
            if (s1 == nSubs-1) {
                delete [] masked_matrices2[s2]->matrix;
                masked_matrices2[s2]->matrix = NULL;
            }
            
        } // loop accumulating N-1 brains

        // here we need to normalize the summed brain
        normbuf(acc, voxels2, trs);

        STOP_TIMER(sumbrain_timer);
        
        START_TIMER(correlation_timer, "correlate N-1 vs subject");
        // now do the "correlation" (dot products between z-scored vectors), for this trial, between
        // all voxel pairs, subject 1 ("buf1") vs all other subjects ("acc") and store in "corrMat"
        // which will accumulate across the N subjects (since beta on it is 1.0, weights input C[orrmat])
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, voxels1, voxels2, trs, 1.0,
                    buf1, trs, acc, trs, 1.0, corrMat, voxels1);

        STOP_TIMER(correlation_timer);
        
        ++completed_correlations;

        cout << "done correlating all other subjects with subject"  << s1 << endl;
        //cout << "[ running global average correlation = " << matmean(corrMat, completed_correlations) << " ]" << endl;
        
        STOP_TIMER(subject_timer);
        
    } // loop acccumulating correlation matrix
    free(buf1);
    free(acc);
    
    divmat(corrMat, voxels1*voxels2, completed_correlations);
    //cout << "[ global average correlation for this trial = " << matmean(corrMat, correlations) << " ]" << endl << flush;
    
    START_TIMER(save_matrix, "save correlation matrix");

    WriteCorrMatToHDF5(voxels1, voxels2, corrMat, output_file);
    
    STOP_TIMER(save_matrix);
    
    free(corrMat);
    
    STOP_TIMER(total_timer);
}

#endif
