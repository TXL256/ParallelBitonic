#include <algorithm>
#include <bit>
#include <bitset>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits.h>

#include "../include/utils.h"
#include "../include/base_par.cuh"

//inplace
//start inclusive, end exclusive
//direction: 0=descending, 1=ascending
void comp_swap(int * data, int i1, int i2, bool direction){ 
    if (!direction && data[i1]<data[i2] || direction && data[i1]>data[i2]){
        std::swap(data[i1], data[i2]);
    }
}

//inplace
void bitonic_merge(int * data, int start_i, int end_i, bool direction){
    int section_len = end_i-start_i;
    if (section_len <= 1) {return;}
    int mid_i = (start_i + end_i) / 2;
    for (int i = start_i; i < mid_i; i++){
        comp_swap(data, i, i+(section_len/2), direction);
    }
    bitonic_merge(data, start_i, mid_i, direction);
    bitonic_merge(data, mid_i, end_i, direction);
}

void bitonic_sort(int * data, int N, int start_i, int end_i, bool direction){
    int section_len = end_i-start_i;
    if (section_len <= 1){return;}
    int mid_i = (start_i + end_i) / 2;
    bitonic_sort(data, N, start_i, mid_i, 1);
    bitonic_sort(data, N, mid_i, end_i, 0);
    bitonic_merge(data, start_i, end_i, direction);
}

void serial_implementation(int * input, int * output, int N){
    //TODO: serial implementation
    //make a MAXINT padded array to force array len to be power of 2
    int pow_2_cap = pow(2, ceil(log2(N)));
    int * working = (int*) malloc(sizeof(int) * pow_2_cap);
    for (int i = 0; i < N; i++){
        working[i] = input[i];
    }
    for (int i = N; i < pow_2_cap; i++){
        working[i] = INT_MAX;
    }
    bitonic_sort(working, pow_2_cap, 0, pow_2_cap, 1);
    for (int i = 0; i < N; i++){
        output[i] = working[i];
    }
    free(working);
    return;
}

void control_implementation(int * input, int * output, int N){
    //TODO: control implementation
    for (int i = 0; i < N; i++){
        output[i] = input[i];
    }
    std::sort(output, output+N);
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


    //TODO: sort using imported algorithm
    int * control_sorted = (int*) malloc(sizeof(int) * N);
    control_implementation(data, control_sorted, N);

    //TODO: sort using serial algorithm
    int * serial_sorted = (int*) malloc(sizeof(int) * N);
    serial_implementation(data, serial_sorted, N);

    int * second_parallel_sorted = (int*) malloc(sizeof(int) * N);
    thr_parallel_implementation(data, second_parallel_sorted, N);

    cudaStreamSynchronize(stream);
    float ms;
    cudaEventElapsedTime(&ms, begin, end);
    printf("Elapsed time: %f ms\n", ms);

    if (N <= 64){
        printf("u  c  s  p\n");
        for (int i=0; i<N; i++){
            printf("%-2d %-2d %-2d %-2d\n", data[i], control_sorted[i], serial_sorted[i], second_parallel_sorted[i]);
        }
    } else {
        for (int i=0; i < N; i++){
            if (control_sorted[i] != serial_sorted[i]) {
                printf("ERROR; serial incorrect: %d != %d @ %d\n", control_sorted[i], serial_sorted[i], i);
            }
            if (control_sorted[i] != second_parallel_sorted[i]) {
                printf("ERROR; parallel incorrect: %d != %d @ %d\n", control_sorted[i], second_parallel_sorted[i], i);
            }
        }
    }
    

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);

    //TODO: deallocate resources
    free(data);
    free(control_sorted);
    free(serial_sorted);
    free(second_parallel_sorted);

    return 0;
}