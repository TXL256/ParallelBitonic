
__device__ void thr_comp_swap(int * data, int N, int i1, int i2, bool direction);
__global__ void partial_step(int * data, int N, int phase, int first_step);
__global__ void full_step(int * data, int N, int phase, int first_step, int end_step);
void thr_parallel_implementation(int * input, int * output, int N);

