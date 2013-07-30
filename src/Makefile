include Makefile.rondo
#include Makefile.della
#include Makefile.yida

OBJ = svm.o Preprocessing.o FileProcessing.o MatComputation.o CorrMatAnalysis.o Classification.o LibSVM.o SVMClassification.o SVMPredictor.o Scheduler.o SVMPredictorWithMasks.o Searchlight.o CorrelationVisualization.o main.o

pni_fcma: $(OBJ)
	$(MPICXX) -o $@ $(CXXFLAGS) $(MPIFLAGS) $(LDFLAGS) $^ $(LD_LIBS)

svm.o: svm.h
	$(CXX) -c svm.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

Preprocessing.o: common.h Preprocessing.h Preprocessing.cpp
	$(CXX) -c Preprocessing.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

FileProcessing.o: common.h Preprocessing.h FileProcessing.cpp
	$(CXX) -c FileProcessing.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

MatComputation.o: common.h MatComputation.h MatComputation.cpp
	$(CXX) -c MatComputation.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

CorrMatAnalysis.o: common.h CorrMatAnalysis.h CorrMatAnalysis.cpp
	$(CXX) -c CorrMatAnalysis.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

Classification.o: common.h Classification.h Classification.cpp
	$(CXX) -c Classification.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

Scheduler.o: common.h Scheduler.h Scheduler.cpp
	$(MPICXX) -c Scheduler.cpp -o $@ $(CXXFLAGS) $(MPIFLAGS)

LibSVM.o: common.h LibSVM.h LibSVM.cpp
	$(CXX) -c LibSVM.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

SVMClassification.o: common.h SVMClassification.h SVMClassification.cpp
	$(CXX) -c SVMClassification.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

SVMPredictor.o: common.h SVMPredictor.h SVMPredictor.cpp
	$(CXX) -c SVMPredictor.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

SVMPredictorWithMasks.o: common.h SVMPredictorWithMasks.h SVMPredictorWithMasks.cpp
	$(CXX) -c SVMPredictorWithMasks.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

Searchlight.o: common.h Searchlight.h Searchlight.cpp
	$(CXX) -c Searchlight.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

CorrelationVisualization.o: common.h CorrelationVisualization.h CorrelationVisualization.cpp
	$(CXX) -c CorrelationVisualization.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

main.o: common.h main.cpp
	$(MPICXX) -c main.cpp -o $@ $(CXXFLAGS) $(MPIFLAGS)

PROGS=pni_fcma

clean:
	rm -f $(PROGS) $(OBJ) *~

