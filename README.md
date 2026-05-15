# ParallelBitonic
A CUDA implementation of bitonic sort based on "Fast In-Place Sorting with CUDA Based on Bitonic Sort" by Hagen Peters et. al.

To run on a randomized array:
1. run on terminal:
make all
./bin/cuda_project -n <length of array>

To run on a file: 
1. place your file into the "data" folder
    (file must be of the form: first line = length of array, second line = space separated list of numbers in the array to be sorted)
2. run on terminal:
make all
./bin/cuda_project -f ./data/<yourfilename>

There exist a few data files I created myself as a demo.


Files generated using LLM: 
- Makefile
I thought myself above it, but there were no good resources on how to write a makefile more complicated than for two files.