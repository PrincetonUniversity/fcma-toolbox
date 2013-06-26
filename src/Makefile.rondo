GXX=g++44
MPICXX=mpic++ 
MPIFLAGS=-I/opt/pkg/OPENMPI/openmpi-1.6.2/include
CXXFLAGS = -I/usr/lib/gcc/x86_64-redhat-linux6E/4.4.6/include -I/usr/lib/gcc/x86_64-redhat-linux6E/4.4.6/finclude -I/opt/pkg/NIFTICLIB/nifticlib-2.0.0/include -O3 -fopenmp -fPIC -g
WARNINGFLAGS = -Wall -Werror -Wextra #-Wno-unused-but-set-parameter
LDFLAGS=-L/opt/pkg/NIFTICLIB/nifticlib-2.0.0/lib -L/opt/pkg/GOTOBLAS2/gotoblas2-1.13-gcc44 -L/usr/lib/gcc/x86_64-redhat-linux6E/4.4.6
LD_LIBS = -lm -lgoto2 -lz -lniftiio -lznz -lgfortran #-lstdc++ -lgomp
OBJ = svm.o Preprocessing.o FileProcessing.o MatComputation.o CorrMatAnalysis.o Classification.o LibSVM.o SVMClassification.o SVMPredictor.o Scheduler.o SVMPredictorWithMasks.o Searchlight.o main.o

pni_fcma: $(OBJ)
	$(MPICXX) -o $@ $(CXXFLAGS) $(MPIFLAGS) $(LDFLAGS) $^ $(LD_LIBS)

svm.o: svm.h
	$(GXX) -c svm.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

Preprocessing.o: common.h Preprocessing.h
	$(GXX) -c Preprocessing.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

FileProcessing.o: common.h Preprocessing.h
	$(GXX) -c FileProcessing.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

MatComputation.o: common.h MatComputation.h
	$(GXX) -c MatComputation.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

CorrMatAnalysis.o: common.h CorrMatAnalysis.h
	$(GXX) -c CorrMatAnalysis.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

Classification.o: common.h Classification.h
	$(GXX) -c Classification.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

Scheduler.o: common.h Scheduler.h
	$(MPICXX) -c Scheduler.cpp -o $@ $(CXXFLAGS) $(MPIFLAGS)

LibSVM.o: common.h LibSVM.h
	$(GXX) -c LibSVM.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

SVMClassification.o: common.h SVMClassification.h
	$(GXX) -c SVMClassification.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

SVMPredictor.o: common.h SVMPredictor.h
	$(GXX) -c SVMPredictor.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

SVMPredictorWithMasks.o: common.h SVMPredictorWithMasks.h
	$(GXX) -c SVMPredictorWithMasks.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

Searchlight.o: common.h Searchlight.h
	$(GXX) -c Searchlight.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

main.o: common.h
	$(MPICXX) -c main.cpp -o $@ $(CXXFLAGS) $(MPIFLAGS)

PROGS=pni_fcma

clean:
	rm -f $(PROGS) $(OBJ) *~

