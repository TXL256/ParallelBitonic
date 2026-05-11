#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <bit>
#include <bitset>
#include <limits.h>

#include "../include/base_par.cuh"

const int ln_per_thr = 16; //2^4, lanes per thread; the number of items one thread is able to hold in cache
const int thr_per_blk = 256; //2^8
const int ln_per_blk = ln_per_thr * thr_per_blk;
const int thr_depth = 4;
const int blk_depth = 8;

__device__ void full_thr_dive(int * data, int N, int phase, int blk_first_step, int thr_first_step){

}

__device__ void part_thr_dive(int * data, int N, int phase, int blk_first_step, int thr_first_step){

}

__global__ void full_blk_dive(int * data, int N, int phase, int first_step){
    //full block dive property: direction is always uniform across a block

}

__global__ void part_blk_dive(int * data, int N, int phase, int first_step){
    //partial block dive property: block spacing is always 1
}

void blk_parallel_implementation(int * input, int * output, int N){
    //TODO: sort using parallel algorithm
    //var definitions
    int padded_N = pow(2, ceil(log2(N)));
    if (padded_N < ln_per_blk){padded_N = ln_per_blk;}
    int * working_arr = (int *) malloc(sizeof(int) * padded_N);
    //device mem allocate
    int * d_working_arr;
    cudaMalloc(&d_working_arr, sizeof(int) * padded_N);
    //memory transfer
    for (int i = 0; i < N; i++){working_arr[i] = input[i];}
    for (int i = N; i < padded_N; i++){working_arr[i] = INT_MAX;}
    cudaMemcpy(d_working_arr, working_arr, sizeof(int) * padded_N, cudaMemcpyHostToDevice);
    //TODO: kernel launch, change phase bound back to log2(N)!!!!!!
    for (int phase = 0; phase < log2(padded_N); phase++){
        for (int step = phase; step >= blk_depth-1; step -= blk_depth){
            full_blk_dive<<<(padded_N/ln_per_blk), thr_per_blk>>>(d_working_arr, padded_N, phase, step);
        }
        if ((phase+1)%blk_depth!=0){
            part_blk_dive<<<(padded_N/ln_per_blk), thr_per_blk>>>(d_working_arr, padded_N, phase, phase%blk_depth);
        }
    }

    //memory transfer
    cudaMemcpy(working_arr, d_working_arr, sizeof(int) * padded_N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++){output[i] = working_arr[i];}

    //resource deallocation
    cudaFree(d_working_arr);
    free(working_arr);
}