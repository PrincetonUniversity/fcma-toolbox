/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include <iostream>  // nullptr
#include "LibSVM.h"

void print_null(const char*) {}  // for SVM print quietly

/*******************************************
set the SVM paramters, most of them are set by default
input: the kernel type
output: the SVM parameter struct defined by libSVM
********************************************/
SVMParameter* SetSVMParameter(int kernel_type) {
  SVMParameter* param = new SVMParameter();
  param->svm_type = C_SVC;  // NU_SVC for activation feature selection and
                            // classification C_SVC for correlation
  param->kernel_type =
      kernel_type;  // 0 for linear, 2 for RBF, 4 for precomputed
  param->degree = 3;
  param->gamma = 0;  // 1/num_features
  param->coef0 = 0;
  param->nu = 0.5;
  param->cache_size = 10000;
  param->C = 1;
  param->eps = 1e-3;
  param->p = 0.1;
  param->shrinking = 0;
  param->probability = 0;
  param->nr_weight = 0;
  param->weight_label = nullptr;
  param->weight = nullptr;
  return param;
}
