#include "common.h"
#include "svm.h"

//libsvm
typedef struct svm_problem SVMProblem;
typedef struct svm_parameter SVMParameter;
typedef struct svm_node SVMNode;

void print_null(const char* s);
SVMParameter* SetSVMParameter(int kernel_type);
