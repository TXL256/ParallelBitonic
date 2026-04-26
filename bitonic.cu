#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "util.h"


int * serial_implementation(int * input, int N){
    //TODO: serial implementation
}

int * control_implementation(int * input, int N){
    //TODO: control implementation
}

//in: name of file to read an array of ints from
//out: correctness evaluation for serial and parallel implementations compared to a imported sorting alg as a control
//out: time comparison between imported sorter, serial sorter, and parallel sorter.
int main(int argc, char ** argv) {
    
    int N;

    assert(argc == 2);
    int * data = read_file(argv[1], &N);

    cudaStream_t stream;
    cudaEvent_t begin, end;
    cudaStreamCreate(&stream);
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    int * parallel_sorted;

    //TODO: sort using imported algorithm

    //TODO: sort using serial algorithm

    //TODO: sort using parallel algorithm
    //TODO: var definitions
    //TODO: device mem allocation
    //TODO: memory transfer
    //TODO: kernel launch
    //TODO: memory transfer

    cudaStreamSynchronize(stream);
    float ms;
    cudaEventElapsedTime(&ms, begin, end);
    printf("Elapsed time: %f ms\n", ms);

    //TODO: deallocate resources

    int * control_sorted = control_implementation(data, N);
    int * serial_sorted = serial_implementation(data, N);
    for (int i=0; i < N; i++){
        if (control_sorted[i] != serial_sorted[i]) {
            printf("ERROR; serial incorrect: %d != %d @ %d\n", control_sorted[i], serial_sorted[i], i);
        }
        if (control_sorted[i] != parallel_sorted[i]) {
            printf("ERROR; parallel incorrect: %d != %d @ %d\n", control_sorted[i], parallel_sorted[i], i);
        }
    }
    

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);

    free(data);

    return 0;
}