/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#ifndef _LIBSVM_H
#define _LIBSVM_H

//#include "common.h"
#include "svm.h"

//libsvm
typedef struct svm_problem SVMProblem;
typedef struct svm_parameter SVMParameter;
typedef struct svm_node SVMNode;
typedef signed char schar;

#ifdef __cplusplus
extern "C" {
#endif
enum { C_SVC };	/* svm_type */ // only C_SVC is needed for now
enum { LINEAR, PRECOMPUTED }; /* kernel_type */
#ifdef __cplusplus
}
#endif

void print_null(const char* s);
SVMParameter* SetSVMParameter(int kernel_type);
#endif /* _LIBSVM_H */
