# include Make.user here
# edit to match your environment
include Make.user

# the above included Makefile should supply:
# BLASINCLUDE, BLASLIBINC, BLASLDFLAGS, BLASCFLAGS
# and optionally OTHER_INCLUDES

# can override anything with ?= as well
MPICC ?= mpic++
NIFTILOC ?= ../deps/nifticlib-2.0.0
WARNFLAGS ?= -Wall -Wextra
OPTIMFLAGS ?= -O3 -fopenmp -pthread -ansi_alias
EXECUTABLE ?= pni_fcma

LDFLAGS = $(BLASLDFLAGS) $(BLASLIBINC) -L$(NIFTILOC)/lib -L$(HOME)/code/fcma-toolbox/src_th/multi_depend_th -lniftiio -lznz -lz
INCLUDES = $(BLASINCLUDE) $(OTHER_INCLUDES) -I$(NIFTILOC)/include
CFLAGS = -std=c++0x -c -fPIC -g $(WARNFLAGS) $(OPTIMFLAGS) $(INCLUDES) $(BLASCFLAGS)# -D__MEASURE_TIME__# -D__MIIC__

SOURCES = main.cpp svm.cpp Preprocessing.cpp FileProcessing.cpp MatComputation.cpp CorrMatAnalysis.cpp Classification.cpp LibSVM.cpp SVMClassification.cpp SVMPredictor.cpp Scheduler.cpp SVMPredictorWithMasks.cpp Searchlight.cpp CorrelationVisualization.cpp ErrorHandling.cpp FisherScoring.cpp Marginal_Scheduler.cpp VoxelwiseAnalysis.cpp

OBJECTS = $(SOURCES:.cpp=.o)
HEADERS = common.h $(SOURCES:.cpp=.h)

CFLAGS_MIC = -mmic $(CFLAGS)
OBJECTS_MIC = $(SOURCES:.cpp=.o.MIC)
BLASLIBINC_MIC = $(BLASLIBINC:intel64=mic)
LDFLAGS_MIC = -mmic $(BLASLDFLAGS) $(BLASLIBINC_MIC)

TARGETS ?= pni_fcma $(OTHER_TARGETS)

all: $(SOURCES) $(TARGETS)

clean:
	rm -f *.o *.MIC $(EXECUTABLE) *~


$(EXECUTABLE): $(OBJECTS)
	@echo "************ building $@ **************"
	$(MPICC) $(OBJECTS) $(LDFLAGS) -o $@
	@echo "************ built    $@ **************"

.cpp.o:
	$(MPICC) $(CFLAGS) $< -o $@

$(EXECUTABLE).MIC: $(OBJECTS_MIC)
	@echo "************ building $@ **************"
	$(MPICC) $(OBJECTS_MIC) $(LDFLAGS_MIC) -o $@
	@echo "************ built    $@ **************"

%.o.MIC: %.cpp
	$(MPICC) $(CFLAGS_MIC) $< -o $@

niftilib:
	@echo "************ building $@ **************"
	(cd $(NIFTILOC); $(MAKE) all;)
	@echo "************ built    $@ **************"

niftilib_clean:
	@echo "********* cleaning niftilib ***********"
	(cd $(NIFTILOC); $(MAKE) clean_all;)
	@echo "********* cleaned  niftilib ***********"

deps: niftilib
	@echo "********* built dependencies  *********"

alldeps: deps all
	@echo "**** built all + dependencies  *****"

depsclean: niftilib_clean
	@echo "**** cleaned all dependencies  *******"

alldepsclean: depsclean clean
	@echo "**** cleaned all + dependencies ******"

distclean: alldepsclean
dist: alldeps

