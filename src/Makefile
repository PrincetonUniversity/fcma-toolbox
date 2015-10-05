# include Make.user here
# edit to match your environment
include Make.user

# the above included Makefile should supply:
# BLASINCLUDE, BLASLIBINC, BLASLDFLAGS, BLASCFLAGS
# and optionally OTHER_INCLUDES and OTHER_LDFLAGS

# can override anything with ?= as well
MPICC ?= mpic++
NIFTILOC ?= ../deps/nifticlib-2.0.0
BASE_CFLAGS ?= -std=c++11 -c -fPIC -g -Wall -Wextra
OMP_FLAGS ?= -fopenmp -pthread
EXECUTABLE ?= pni_fcma
EXEDBG ?= $(EXECUTABLE)_debug
EXEPRO ?= $(EXECUTABLE)_profile

OPTFLAGS ?= -O3 -ansi_alias -DNDEBUG
PROFLAGS ?= -D__MEASURE_TIME__

LDFLAGS = $(BLASLDFLAGS) $(BLASLIBINC) -lSegFault -L$(NIFTILOC)/lib -lniftiio -lznz -lz $(OTHER_LDFLAGS)
INCLUDES = $(BLASINCLUDE) $(OTHER_INCLUDES) -I$(NIFTILOC)/include

CFLAGS = $(BASE_CFLAGS) $(OMP_FLAGS) $(INCLUDES) $(BLASCFLAGS)
CFLAGS_OPT = $(OPTFLAGS) $(CFLAGS)
CFLAGS_PRO = $(OPTFLAGS) $(PROFLAGS) $(CFLAGS)

SOURCES = main.cc Preprocessing.cc FileProcessing.cc MatComputation.cc CustomizedMatrixMultiply.cc CorrMatAnalysis.cc Classification.cc LibSVM.cc SVMClassification.cc SVMPredictor.cc Scheduler.cc SVMPredictorWithMasks.cc Searchlight.cc CorrelationVisualization.cc ErrorHandling.cc FisherScoring.cc Marginal_Scheduler.cc VoxelwiseAnalysis.cc

OBJECTS = $(SOURCES:.cc=.o)
OBJDBG = $(SOURCES:.cc=_debug.o)
OBJPRO = $(SOURCES:.cc=_profile.o)

HEADERS = common.h $(SOURCES:.cc=.h)

CFLAGS_MIC = -mmic $(CFLAGS)
CFLAGS_OPT_MIC = $(OPTFLAGS) $(CFLAGS_MIC)
CFLAGS_PRO_MIC = $(PROFLAGS) $(CFLAGS_MIC)
OBJECTS_MIC = $(OBJECTS:.o=.o.MIC)
OBJDBG_MIC = $(OBJDBG:.o=.o.MIC)
OBJPRO_MIC = $(OBJPRO:.o=.o.MIC)
BLASLIBINC_MIC = $(BLASLIBINC:intel64=mic)
LDFLAGS_MIC = -mmic $(BLASLDFLAGS) $(BLASLIBINC_MIC) $(OTHER_LDFLAGS_MIC)
EXEMIC = $(EXECUTABLE).MIC
EXEDBG_MIC = $(EXEDBG).MIC
EXEPRO_MIC = $(EXEPRO).MIC

all: world world_mic

release_all: release release_mic

world: release debug profile
release: $(SOURCES) $(EXECUTABLE)
debug: $(SOURCES) $(EXEDBG)
profile: $(SOURCES) $(EXEPRO)

world_mic: release_mic debug_mic profile_mic
release_mic: $(SOURCES) $(EXEMIC)
debug_mic: $(SOURCES) $(EXEDBG_MIC)
profile_mic: $(SOURCES) $(EXEPRO_MIC)

clean:
	rm -f *.o *~ *.optrpt *.o.MIC $(EXECUTABLE) $(EXEDBG) $(EXEPRO) $(EXEMIC) $(EXEDBG_MIC) $(EXEPRO_MIC)


$(EXECUTABLE): $(OBJECTS)
	@echo "************ building $@ **************"
	$(MPICC) $(OBJECTS) $(LDFLAGS) -o $@
	@echo "************ built    $@ **************"

$(EXEDBG): $(OBJDBG)
	@echo "************ building $@ **************"
	$(MPICC) $(OBJDBG) $(LDFLAGS) -o $@
	@echo "************ built    $@ **************"

$(EXEPRO): $(OBJPRO)
	@echo "************ building $@ **************"
	$(MPICC) $(OBJPRO) $(LDFLAGS) -o $@
	@echo "************ built    $@ **************"

$(EXEMIC): $(OBJECTS_MIC)
	@echo "************ building $@ **************"
	$(MPICC) $(OBJECTS_MIC) $(LDFLAGS_MIC) -o $@
	@echo "************ built    $@ **************"

$(EXEDBG_MIC): $(OBJDBG_MIC)
	@echo "************ building $@ **************"
	$(MPICC) $(OBJDBG_MIC) $(LDFLAGS_MIC) -o $@
	@echo "************ built    $@ **************"

$(EXEPRO_MIC): $(OBJPRO_MIC)
	@echo "************ building $@ **************"
	$(MPICC) $(OBJPRO_MIC) $(LDFLAGS_MIC) -o $@
	@echo "************ built    $@ **************"

%.o: %.cc
	$(MPICC) $(CFLAGS_OPT) $< -o $@

%_profile.o: %.cc
	$(MPICC) $(CFLAGS_PRO) $< -o $@

%_debug.o: %.cc
	$(MPICC) $(CFLAGS) $< -o $@

%.o.MIC: %.cc
	$(MPICC) $(CFLAGS_OPT_MIC) $< -o $@

%_debug.o.MIC: %.cc
	$(MPICC) $(CFLAGS_MIC) $< -o $@

_profile.o.MIC: %.cc
	$(MPICC) $(CFLAGS_PRO_MIC) $< -o $@

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

