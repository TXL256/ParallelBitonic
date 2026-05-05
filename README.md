# ParallelBitonic
A CUDA implementation of bitonic sort based on "Fast In-Place Sorting with CUDA Based on Bitonic Sort" by Hagen Peters et. al.

To use: 
1. place your file into the "data" folder
    (file must be of the form: first line = length of array, second line = space separated list of numbers in the array to be sorted)
2. run on terminal:
make all
./bitonic_opt ./data/<yourfilename>
