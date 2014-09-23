# include Make.user here
# edit to match your environment
include Make.user

# the above included Makefile should supply:
# BLASINCLUDE, BLASLIBINC, BLASLDFLAGS, BLASCFLAGS
# and optionally OTHER_INCLUDES

# can override anything with ?= as well
MPICC ?= mpicc
NIFTILOC ?= ../deps/nifticlib-2.0.0
WARNFLAGS ?= -Wall -Wextra
OPTIMFLAGS ?= -O3 -fopenmp
EXECUTABLE ?= pni_fcma
BUILD_MIC ?= 0

LDFLAGS = $(BLASLDFLAGS) $(BLASLIBINC) -L$(NIFTILOC)/lib -lniftiio -lznz -lz
INCLUDES = $(BLASINCLUDE) $(OTHER_INCLUDES) -I$(NIFTILOC)/include
CFLAGS = -c -fPIC -g $(WARNFLAGS) $(OPTIMFLAGS) $(INCLUDES) $(BLASCFLAGS)

SOURCES = main.cpp svm.cpp Preprocessing.cpp FileProcessing.cpp MatComputation.cpp CorrMatAnalysis.cpp Classification.cpp LibSVM.cpp SVMClassification.cpp SVMPredictor.cpp Scheduler.cpp SVMPredictorWithMasks.cpp Searchlight.cpp CorrelationVisualization.cpp ErrorHandling.cpp FisherScoring.cpp Marginal_Scheduler.cpp
OBJECTS = $(SOURCES:.cpp=.o)
HEADERS = common.h $(SOURCES:.cpp=.h)

ifeq ($(BUILD_MIC), 1)
OBJECTS_MIC = $(SOURCES:.cpp=.o.MIC)
all: $(SOURCES) $(EXECUTABLE) $(EXECUTABLE).MIC
clean:
	rm -f *.o *.MIC $(EXECUTABLE) *~
else
all: $(SOURCES) $(EXECUTABLE)
clean:
	rm -f *.o $(EXECUTABLE) *~
endif

$(EXECUTABLE): $(OBJECTS)
	$(MPICC) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
	$(MPICC) $(CFLAGS) $< -o $@

$(EXECUTABLE).MIC: $(OBJECTS_MIC)
	mpiicc -mmic $(OBJECTS_MIC) $(BLASLDFLAGS) -L/opt/intel/mkl/lib/mic -o $@

%.o.MIC: %.cpp
	mpiicc -mmic $(CFLAGS) $< -o $@

.PHONY: clean


