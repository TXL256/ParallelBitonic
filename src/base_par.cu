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

//1=ascending
__device__ void thr_comp_swap(int * data, int N, int i1, int i2, bool direction){
    if (!direction && data[i1]<data[i2] || direction && data[i1]>data[i2]){
        int temp = data[i1];
        data[i1] = data[i2];
        data[i2] = temp;
    }
}

__global__ void partial_step(int * data, int N, int phase, int first_step){
    //key property of partial steps: threads operate on continuous areas
    //TODO: test
    int thr_start_i = ln_per_thr * (threadIdx.x + blockDim.x * blockIdx.x); //inclusive
    int thr_end_i = thr_start_i + ln_per_thr; //exclusive
    int dir_sector_size = 1<<(phase+1);
    for (int dir_sector_start = thr_start_i; dir_sector_start < thr_end_i; dir_sector_start+=dir_sector_size){
        bool sort_dir = ((dir_sector_start/dir_sector_size)&1)==0;
        for (int substep = first_step; substep >= 0; substep--){
            int comp_span = 1<<substep;
            for (int minisector_start = dir_sector_start; minisector_start < (dir_sector_start+dir_sector_size) && minisector_start < thr_end_i; minisector_start += comp_span*2){
                for (int i = minisector_start; i < (minisector_start+comp_span); i++){
                    thr_comp_swap(data, N, i, i+comp_span, sort_dir);
                }
            }
        }
    }
}

__global__ void full_step(int * data, int N, int phase, int first_step, int end_step){
    //key property: all comps for a thread face the same direction
    //TODO: test
    //make partition
    //if ((threadIdx.x + blockDim.x * blockIdx.x)>=4 || (threadIdx.x + blockDim.x * blockIdx.x)<2){return;}
    int lns_per_sector = 1<<(first_step+1);
    int thr_per_sector = lns_per_sector/ln_per_thr;
    int thr_assigned_sector = (threadIdx.x + blockDim.x * blockIdx.x) / thr_per_sector; 
    int thr_spacing = thr_per_sector; 
    int thr_sector_start = thr_assigned_sector * lns_per_sector; 
    int thr_sector_end = (thr_assigned_sector+1)*lns_per_sector; 
    int thr_sector_id = (threadIdx.x + blockDim.x * blockIdx.x) % thr_per_sector;
    bool sort_dir = ((thr_sector_start/(thr_sector_end-thr_sector_start))&1)==0;
    //engage in sorting
    for (int substep = first_step; substep > end_step; substep--){ //2-0
        int comp_span = 1<<substep;
        for (int minisector_start = thr_sector_start; minisector_start < thr_sector_end; minisector_start += comp_span*2){
            for (int i = minisector_start+thr_sector_id; i < (minisector_start+comp_span); i += thr_spacing){
                thr_comp_swap(data, N, i, i+comp_span, sort_dir);
            }
        }
    }
}

void thr_parallel_implementation(int * input, int * output, int N){
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
    //TODO: kernel launches
    for (int phase = 0; phase < log2(N); phase++){
        for (int first_step = phase; first_step >= thr_depth-1; first_step -= thr_depth){
            //TODO: full step
            full_step<<<(padded_N/ln_per_blk), thr_per_blk>>>(d_working_arr, padded_N, phase, first_step, first_step-thr_depth);
        }
        //TODO: partial step
        if ((phase+1)%thr_depth!=0) {
            partial_step<<<(padded_N/ln_per_blk), thr_per_blk>>>(d_working_arr, padded_N, phase, phase%thr_depth);
        }
    }
    //memory transfer
    cudaMemcpy(working_arr, d_working_arr, sizeof(int) * padded_N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++){output[i] = working_arr[i];}

    //resource deallocation
    cudaFree(d_working_arr);
    free(working_arr);
}