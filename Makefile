CXX := nvcc
OPT_FLAGS := -O3
DEBUG_FLAGS := -G
GENCODE := -gencode arch=compute_70,code=compute_70 -gencode arch=compute_75,code=compute_75

.phony: clean all debug release

all: debug release

release: bitonic_opt

debug: bitonic_debug

clean:
	rm -f bitonic_opt bitonic_debug

bitonic_opt: bitonic.cu util.h
	$(CXX) $(OPT_FLAGS) -o $@ $< $(GENCODE)

bitonic_debug: bitonic.cu util.h
	$(CXX) $(DEBUG_FLAGS) -o $@ $< $(GENCODE)

handin.tar: bitonic.cu
	tar -cvf handin.tar bitonic.cu

.phony: clean
