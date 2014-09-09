#include Makefile.della
#include Makefile.c8
#include Makefile.rondo
#include Makefile.mac
include Makefile.neuro

OBJ = svm.o Preprocessing.o FileProcessing.o MatComputation.o CorrMatAnalysis.o Classification.o LibSVM.o SVMClassification.o SVMPredictor.o Scheduler.o SVMPredictorWithMasks.o Searchlight.o CorrelationVisualization.o ErrorHandling.o FisherScoring.o Marginal_Scheduler.o

OBJ.MIC = svm.MIC.o Preprocessing.MIC.o FileProcessing.MIC.o MatComputation.MIC.o CorrMatAnalysis.MIC.o Classification.MIC.o LibSVM.MIC.o SVMClassification.MIC.o SVMPredictor.MIC.o Scheduler.MIC.o SVMPredictorWithMasks.MIC.o Searchlight.MIC.o CorrelationVisualization.MIC.o ErrorHandling.MIC.o FisherScoring.MIC.o Marginal_Scheduler.MIC.o


all: pni_fcma pni_fcma.MIC

pni_fcma: $(OBJ) main.cpp
	$(MPICXX) -o $@ $(CXXFLAGS) $(MPIFLAGS) $(LDFLAGS) $^ $(LD_LIBS) $(HOST_LD_LIBS)

pni_fcma.MIC: $(OBJ.MIC) main.cpp
	$(MPICXX) -o $@ $(CXXFLAGS) $(MPIFLAGS) $(LDFLAGS) $^ $(LD_LIBS) $(ARCHFLAGS)

shlib: $(OBJ) main.cpp
	$(MPICXX) -shared -o $(SHLIB) $(CXXFLAGS) $(MPIFLAGS) $(LDFLAGS) -DNOMAIN $^ $(LD_LIBS)

svm.o: svm.h
	$(CXX) -c svm.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

svm.MIC.o: svm.h
	$(CXX) -c svm.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS) $(ARCHFLAGS)

#svm.o: linear_svm.h
#	$(CXX) -c linear_svm.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

Preprocessing.o: common.h Preprocessing.h Preprocessing.cpp
	$(CXX) -c Preprocessing.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

Preprocessing.MIC.o: common.h Preprocessing.h Preprocessing.cpp
	$(CXX) -c Preprocessing.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS) $(ARCHFLAGS) -vec-report=3

FileProcessing.o: common.h Preprocessing.h FileProcessing.cpp
	$(CXX) -c FileProcessing.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

FileProcessing.MIC.o: common.h Preprocessing.h FileProcessing.cpp
	$(CXX) -c FileProcessing.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS) $(ARCHFLAGS)

MatComputation.o: common.h MatComputation.h MatComputation.cpp
	$(CXX) -c MatComputation.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

MatComputation.MIC.o: common.h MatComputation.h MatComputation.cpp
	$(CXX) -c MatComputation.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS) $(ARCHFLAGS)

CorrMatAnalysis.o: common.h CorrMatAnalysis.h CorrMatAnalysis.cpp
	$(CXX) -c CorrMatAnalysis.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

CorrMatAnalysis.MIC.o: common.h CorrMatAnalysis.h CorrMatAnalysis.cpp
	$(CXX) -c CorrMatAnalysis.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS) $(ARCHFLAGS)

Classification.o: common.h Classification.h Classification.cpp
	$(CXX) -c Classification.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

Classification.MIC.o: common.h Classification.h Classification.cpp
	$(CXX) -c Classification.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS) $(ARCHFLAGS)

Scheduler.o: common.h Scheduler.h Scheduler.cpp
	$(MPICXX) -c Scheduler.cpp -o $@ $(CXXFLAGS) $(MPIFLAGS)

Scheduler.MIC.o: common.h Scheduler.h Scheduler.cpp
	$(MPICXX) -c Scheduler.cpp -o $@ $(CXXFLAGS) $(MPIFLAGS) $(ARCHFLAGS)

Marginal_Scheduler.o: common.h Marginal_Scheduler.h Marginal_Scheduler.cpp
	$(MPICXX) -c Marginal_Scheduler.cpp -o $@ $(CXXFLAGS) $(MPIFLAGS)

Marginal_Scheduler.MIC.o: common.h Marginal_Scheduler.h Marginal_Scheduler.cpp
	$(MPICXX) -c Marginal_Scheduler.cpp -o $@ $(CXXFLAGS) $(MPIFLAGS) $(ARCHFLAGS)

LibSVM.o: common.h LibSVM.h LibSVM.cpp
	$(CXX) -c LibSVM.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

LibSVM.MIC.o: common.h LibSVM.h LibSVM.cpp
	$(CXX) -c LibSVM.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS) $(ARCHFLAGS)

SVMClassification.o: common.h SVMClassification.h SVMClassification.cpp
	$(CXX) -c SVMClassification.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

SVMClassification.MIC.o: common.h SVMClassification.h SVMClassification.cpp
	$(CXX) -c SVMClassification.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS) $(ARCHFLAGS)

SVMPredictor.o: common.h SVMPredictor.h SVMPredictor.cpp
	$(CXX) -c SVMPredictor.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

SVMPredictor.MIC.o: common.h SVMPredictor.h SVMPredictor.cpp
	$(CXX) -c SVMPredictor.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS) $(ARCHFLAGS)

SVMPredictorWithMasks.o: common.h SVMPredictorWithMasks.h SVMPredictorWithMasks.cpp
	$(CXX) -c SVMPredictorWithMasks.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

SVMPredictorWithMasks.MIC.o: common.h SVMPredictorWithMasks.h SVMPredictorWithMasks.cpp
	$(CXX) -c SVMPredictorWithMasks.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS) $(ARCHFLAGS)

Searchlight.o: common.h Searchlight.h Searchlight.cpp
	$(CXX) -c Searchlight.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

Searchlight.MIC.o: common.h Searchlight.h Searchlight.cpp
	$(CXX) -c Searchlight.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS) $(ARCHFLAGS)

CorrelationVisualization.o: common.h CorrelationVisualization.h CorrelationVisualization.cpp
	$(CXX) -c CorrelationVisualization.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

CorrelationVisualization.MIC.o: common.h CorrelationVisualization.h CorrelationVisualization.cpp
	$(CXX) -c CorrelationVisualization.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS) $(ARCHFLAGS)

ErrorHandling.o: common.h ErrorHandling.h ErrorHandling.cpp
	$(MPICXX) -c ErrorHandling.cpp -o $@ $(CXXFLAGS) $(MPIFLAGS) #$(WARNINGFLAGS)

ErrorHandling.MIC.o: common.h ErrorHandling.h ErrorHandling.cpp
	$(MPICXX) -c ErrorHandling.cpp -o $@ $(CXXFLAGS) $(MPIFLAGS) $(ARCHFLAGS) #$(WARNINGFLAGS)

FisherScoring.o: common.h FisherScoring.h FisherScoring.cpp
	$(CXX) -c FisherScoring.cpp -o $@ $(CXXFLAGS) $(WARNINGFLAGS)

FisherScoring.MIC.o: common.h FisherScoring.h FisherScoring.cpp
	$(CXX) -c FisherScoring.cpp -o $@ $(CXXFLAGS) $(ARCHFLAGS) $(WARNINGFLAGS) -vec-report=3

PROGS=pni_fcma pni_fcma.MIC

clean:
	rm -f $(PROGS) $(OBJ) $(OBJ.MIC) $(SHLIB) *~

