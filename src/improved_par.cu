#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <bit>
#include <bitset>
#include <limits.h>

#include "../include/utils.h"

const int ln_per_thr = 16; //2^4, lanes per thread; the number of items one thread is able to hold in cache
const int thr_per_blk = 256; //2^8
const int ln_per_blk = ln_per_thr * thr_per_blk;
const int thr_depth = 4;
const int blk_depth = 8;

__device__ void thr_comp_swap2(int * data, int N, int i1, int i2, bool direction){
    if (i1 >= N || i2 >= N){
        printf("Err: tried to compare outside array bounds");
    }
    if (!direction && data[i1]<data[i2] || direction && data[i1]>data[i2]){
        int temp = data[i1];
        data[i1] = data[i2];
        data[i2] = temp;
    }
}

//sorting direction is uniform across a thread.
__device__ void full_thr_dive(int * blk_data, int phase, int blk_first_step, int thr_first_step){
    //TODO: test
    int blk_last_step = blk_first_step-blk_depth;
    int global_thr_id = threadIdx.x + blockDim.x * blockIdx.x;
    int lns_per_sector = 1<<(thr_first_step-blk_last_step);
    int thr_per_sector = lns_per_sector/ln_per_thr;
    int thr_assigned_sector = threadIdx.x/thr_per_sector;
    int thr_spacing = 1<<(thr_first_step-(blk_first_step-blk_depth));
    int thr_sector_start = thr_assigned_sector * lns_per_sector; 
    int thr_sector_end = (thr_assigned_sector+1)*lns_per_sector; 
    int thr_sector_id = threadIdx.x % thr_per_sector;
    int dir_sector_size = 1<<(phase+1);
    bool sort_dir = (((global_thr_id*ln_per_thr)/dir_sector_size)%2)==0;
    for (int substep = thr_first_step; substep > thr_first_step-thr_depth; substep--){
        int comp_span = 1<<(substep-(blk_first_step-blk_depth)-1);
        for (int minisector_start = thr_sector_start; minisector_start < thr_sector_end; minisector_start += 2*comp_span){
            for (int i = minisector_start+thr_sector_id; i < (minisector_start+comp_span); i+=thr_spacing){
                thr_comp_swap2(blk_data, ln_per_blk, i, i+comp_span, sort_dir);
            }
        }
    }
}

//only ever used by partial block dives
//block AND thread spacing is 1: really simple partitioning, but sorting direction may change across a thread
//remember: blk_data array only has access to part of the dataset
__device__ void part_thr_dive(int * blk_data, int phase, int blk_first_step, int thr_first_step){
    //TODO: test
    int global_thr_id = threadIdx.x + blockDim.x * blockIdx.x;
    int thr_start_i = ln_per_thr * global_thr_id; //inclusive
    int thr_end_i = thr_start_i + ln_per_thr; //exclusive
    int dir_sector_size = 1<<(phase+1);
    for (int dir_sector_start = thr_start_i; dir_sector_start < thr_end_i; dir_sector_start+=dir_sector_size){
        bool sort_dir = ((dir_sector_start/dir_sector_size)&1)==0;
        for (int substep = thr_first_step; substep >= 0; substep--){
            int comp_span = 1<<substep;
            for (int minisector_start = dir_sector_start; minisector_start < (dir_sector_start+dir_sector_size) && minisector_start < thr_end_i; minisector_start += comp_span*2){
                for (int i = minisector_start; i < (minisector_start+comp_span); i++){
                    thr_comp_swap2(blk_data, ln_per_blk, i%ln_per_blk, (i+comp_span)%ln_per_blk, sort_dir);
                }
            }
        }
    }
}

//full block dive property: direction is always uniform across a block
__global__ void full_blk_dive(int * data, int N, int phase, int first_step){
    //TODO: test
    int dir_sector_size = 1<<(phase+1);
    int blk_per_dir_sector = dir_sector_size/ln_per_blk; //equal to block spacing
    int blk_dir_sector_id = blockIdx.x % blk_per_dir_sector;
    int blk_dir_sector_start = (blockIdx.x/blk_per_dir_sector) * dir_sector_size;
    
    //transfer partition to shared mem
    __shared__ int blk_data[ln_per_blk];
    for (int i = 0; i < ln_per_thr; i++){
        int global_addr = blk_dir_sector_start+blk_dir_sector_id+(blk_per_dir_sector*(i+threadIdx.x*ln_per_thr));
        blk_data[i+threadIdx.x*ln_per_thr] = data[global_addr];
    }

    //do thread dives
    for (int dive_step = first_step; dive_step > first_step-blk_depth; dive_step-=thr_depth){
        full_thr_dive(blk_data, phase, first_step, dive_step);
    }

    //transfer partition back to global mem
    for (int i = 0; i < ln_per_thr; i++){
        int global_addr = blk_dir_sector_start+blk_dir_sector_id+(blk_per_dir_sector*(i+threadIdx.x*ln_per_thr));
        data[global_addr] = blk_data[i+threadIdx.x*ln_per_thr];
    }
}

//partial block dive property: block spacing is always 1
__global__ void part_blk_dive(int * data, int N, int phase, int first_step){
    //TODO: test
    //transfer partition of data to shared mem
    __shared__ int blk_data[ln_per_blk];
    int global_thr_id = threadIdx.x + blockDim.x + blockIdx.x;
    for (int i = 0; i < ln_per_thr; i++){
        blk_data[threadIdx.x*ln_per_thr + i] = data[global_thr_id*ln_per_thr + i];
    }

    //TODO: do thread dives
    for (int dive_step = first_step; dive_step >= thr_depth-1; dive_step-=thr_depth){
        full_thr_dive(blk_data, phase, first_step, dive_step);
    }
    if ((first_step+1)%thr_depth != 0){
        part_thr_dive(blk_data, phase, first_step, first_step%thr_depth);
    }

    //transfer processed partition back to global mem
    for (int i = 0; i < ln_per_thr; i++){
        data[global_thr_id*ln_per_thr + i] = blk_data[threadIdx.x*ln_per_thr + i];
    }

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