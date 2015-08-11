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

LDFLAGS = $(BLASLDFLAGS) $(BLASLIBINC) -L$(NIFTILOC)/lib -lniftiio -lznz -lz
INCLUDES = $(BLASINCLUDE) $(OTHER_INCLUDES) -I$(NIFTILOC)/include
CFLAGS = -std=c++11 -c -fPIC -g $(WARNFLAGS) $(OPTIMFLAGS) $(INCLUDES) $(BLASCFLAGS)# -D__MEASURE_TIME__

SOURCES = main.cc svm.cc Preprocessing.cc FileProcessing.cc MatComputation.cc CustomizedMatrixMultiply.cc CorrMatAnalysis.cc Classification.cc LibSVM.cc SVMClassification.cc SVMPredictor.cc Scheduler.cc SVMPredictorWithMasks.cc Searchlight.cc CorrelationVisualization.cc ErrorHandling.cc FisherScoring.cc Marginal_Scheduler.cc VoxelwiseAnalysis.cc new_svm.cc

OBJECTS = $(SOURCES:.cc=.o)
HEADERS = common.h $(SOURCES:.cc=.h)

CFLAGS_MIC = -mmic $(CFLAGS)
OBJECTS_MIC = $(SOURCES:.cc=.o.MIC)
BLASLIBINC_MIC = $(BLASLIBINC:intel64=mic)
LDFLAGS_MIC = -mmic $(BLASLDFLAGS) $(BLASLIBINC_MIC)

TARGETS ?= pni_fcma $(OTHER_TARGETS)

all: $(SOURCES) $(TARGETS)

clean:
	rm -f *.o *.MIC $(EXECUTABLE) *~ *.optrpt


$(EXECUTABLE): $(OBJECTS)
	@echo "************ building $@ **************"
	$(MPICC) $(OBJECTS) $(LDFLAGS) -o $@
	@echo "************ built    $@ **************"

.cc.o:
	$(MPICC) $(CFLAGS) $< -o $@

$(EXECUTABLE).MIC: $(OBJECTS_MIC)
	@echo "************ building $@ **************"
	$(MPICC) $(OBJECTS_MIC) $(LDFLAGS_MIC) -o $@
	@echo "************ built    $@ **************"

%.o.MIC: %.cc
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

