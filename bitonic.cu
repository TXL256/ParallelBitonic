#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <bit>
#include <bitset>
#include <limits.h>

#include "util.h"

const int ln_per_thr = 16; //2^4, lanes per thread; the number of items one thread is able to hold in cache
const int thr_per_blk = 256; //2^8
const int ln_per_blk = ln_per_thr * thr_per_blk;
const int thr_depth = 4;
const int blk_depth = 8;

//1=ascending
__device__ void thr_comp_swap(int * data, int N, int i1, int i2, bool direction){
    if (!direction && i1<i2 || direction && i1>i2){
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
        bool sort_dir = (bool) ((dir_sector_start/dir_sector_size)+1)%2;
        for (int substep = first_step; first_step >= 0; first_step--){
            int comp_span = 1<<substep;
            for (int minisector_start = dir_sector_start; minisector_start < (dir_sector_start+dir_sector_size); minisector_start += comp_span*2){
                for (int i = minisector_start; i < (minisector_start+comp_span); i++){
                    thr_comp_swap(data, N, i, i+comp_span, sort_dir);
                }
            }
        }
    }
}

__global__ void full_step(int * data, int N, int phase, int first_step, int end_step){
    //key property: all comps for a thread face the same direction
    //TODO: fill in
    //make partition
    int num_thrs = N/ln_per_thr; //16
    int lns_per_sector = 1<<(first_step+1); //8
    int thr_per_sector = lns_per_sector/ln_per_thr; //2
    int num_sectors = N/lns_per_sector; //8
    int thr_assigned_sector = (threadIdx.x + blockDim.x * blockIdx.x) / thr_per_sector; //1
    int thr_spacing = thr_per_sector; //2
    int thr_sector_start = thr_assigned_sector * lns_per_sector; //8
    int thr_sector_end = (thr_assigned_sector+1)*lns_per_sector; //16
    int thr_sector_id = (threadIdx.x + blockDim.x * blockIdx.x) % thr_per_sector; //1
    bool sort_dir = (bool) ((thr_sector_start / (1<<(phase+1)))+1)%2; //0
    //thr/sector*numsectors=numthrs
    for (int substep = first_step; substep > end_step; substep--){ //2-0
        int comp_span = 1<<substep;
        for (int minisector_start = thr_sector_start; minisector_start < thr_sector_end; minisector_start += comp_span*2){
            for (int i = minisector_start+thr_sector_id; i < (minisector_start+comp_span); i += thr_spacing){
                thr_comp_swap(data, N, i, i+comp_span, sort_dir);
            }
        }
    }
}

void parallel_implementation(int * input, int * output, int N){
    //TODO: sort using parallel algorithm
    //TODO: var definitions
    int padded_N = std::__bit_ceil(N);
    int num_blks = padded_N / ln_per_blk;
    int * working_arr = (int *) malloc(sizeof(int) * padded_N);
    //device mem allocate
    int * d_working_arr;
    cudaMalloc(&d_working_arr, sizeof(int) * padded_N);
    //memory transfer
    for (int i = 0; i < N; i++){working_arr[i] = input[i];}
    for (int i = N; i < padded_N; i++){working_arr[i] = INT_MAX;}
    cudaMemcpy(d_working_arr, working_arr, sizeof(int) * padded_N, cudaMemcpyHostToDevice);
    //TODO: kernel launch
    for (int phase = 0; phase < __clz(N); phase ++){
        for (int first_step = phase; first_step >= thr_depth; first_step -= thr_depth){
            //full step
        }
        //partial step
    }
    //TODO: memory transfer
    cudaMemcpy(working_arr, d_working_arr, sizeof(int) * padded_N, cudaMemcpyDeviceToHost);

    free(working_arr);
}

//inplace
//start inclusive, end exclusive
//direction: 0=descending, 1=ascending
void comp_swap(int * data, int i1, int i2, bool direction){ 
    if (!direction && i1<i2 || direction && i1>i2){
        int temp = data[i1];
        data[i1] = data[i2];
        data[i2] = temp;
    }
}

//inplace
void bitonic_merge(int * data, int start_i, int end_i, bool direction){
    int section_len = end_i-start_i;
    if (section_len <= 1) {return;}
    for (int i = 0; i < section_len/2; i++){
        comp_swap(data, i, i+section_len/2, direction);
    }
    int mid_i = (start_i + end_i) / 2;
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
    int pow_2_cap = std::__bit_ceil(N);
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

    int * parallel_sorted = (int*) malloc(sizeof(int) * N);
    parallel_implementation(data, parallel_sorted, N);

    cudaStreamSynchronize(stream);
    float ms;
    cudaEventElapsedTime(&ms, begin, end);
    printf("Elapsed time: %f ms\n", ms);



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

    //TODO: deallocate resources
    free(data);
    free(control_sorted);
    free(serial_sorted);
    free(parallel_sorted);

    return 0;
}