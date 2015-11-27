# CS420_project

Cannon's algorithm, a distributed parallel matrix-matrix multiplication algorithm is implemented and compared to a naive MMM algorithm. 

Summary of Early Sequential MMM tests: 

1. BLAS multiplication is the fastest at matrix sizes larger than 200x200. 

2. Loop tiling is second fastest, but you have to keep varying blocksize to find the optimal value.

3. Optimal blocksize seems to be as low as 40 and as high as 100. 

4. Use mpirun -np 1 ./naive.exe 1000 1000 1000 1 1 {0,1,2,3,4} {blocksize} {number of threads} to test this. 

Todo: 
* We should try using internal optimized MMM libraries like BLAS and LAPACK and compare the result to our parallelized code. 
* We could also try to profile better using PMPI maybe? The current gettime function just measures the Rank 0 time, which could be misleading. 
