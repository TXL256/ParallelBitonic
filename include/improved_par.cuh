__device__ void thr_comp_swap(int * data, int N, int i1, int i2, bool direction);
__device__ void full_thr_dive(int * data, int N, int phase, int blk_first_step, int thr_first_step);
__device__ void part_thr_dive(int * data, int N, int phase, int blk_first_step, int thr_first_step);
__global__ void full_blk_dive(int * data, int N, int phase, int first_step);
__global__ void part_blk_dive(int * data, int N, int phase, int first_step);
void blk_parallel_implementation(int * input, int * output, int N);