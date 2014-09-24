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

CFLAGS_MIC = -mmic $(CFLAGS)
OBJECTS_MIC = $(SOURCES:.cpp=.o.MIC)
BLASLIBINC_MIC = $(BLASLIBINC:intel64=mic)
LDFLAGS_MIC = -mmic $(BLASLDFLAGS) $(BLASLIBINC_MIC)
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
	$(MPICC) $(OBJECTS_MIC) $(LDFLAGS_MIC) -o $@

%.o.MIC: %.cpp
	$(MPICC) $(CFLAGS_MIC) $< -o $@

.PHONY: clean


