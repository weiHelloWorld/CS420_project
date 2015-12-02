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

The output from batch_script.run is shown below:

Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                12
On-line CPU(s) list:   0-11
Thread(s) per core:    1
Core(s) per socket:    6
Socket(s):             2
NUMA node(s):          2
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 44
Stepping:              2
CPU MHz:               2666.823
BogoMIPS:              5333.20
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              12288K
NUMA node0 CPU(s):     0,2,4,6,8,10
NUMA node1 CPU(s):     1,3,5,7,9,11
 
Running naive.exe with size=1000
Total initialization time is 0.025437
Clocal time is 0.055346
[1000 1000 1000 1 5 4 10] Naive Total Running Time: 0.106525
 
Running naive.exe with size=2000
Total initialization time is 0.072848
Clocal time is 0.176334
[2000 2000 2000 1 5 4 10] Naive Total Running Time: 0.333396
 
Running naive.exe with size=3000
Total initialization time is 0.117013
Clocal time is 0.550725
[3000 3000 3000 1 5 4 10] Naive Total Running Time: 0.843320
 
Running naive.exe with size=4000
Total initialization time is 0.196536
Clocal time is 1.284818
[4000 4000 4000 1 5 4 10] Naive Total Running Time: 1.786551
 
Running naive.exe with size=5000
Total initialization time is 0.311335
Clocal time is 2.477229
[5000 5000 5000 1 5 4 10] Naive Total Running Time: 3.315478
 
Running naive.exe with size=6000
Total initialization time is 0.433675
Clocal time is 4.216763
[6000 6000 6000 1 5 4 10] Naive Total Running Time: 5.406512
 
Running naive.exe with size=7000
Total initialization time is 0.600290
Clocal time is 6.735082
[7000 7000 7000 1 5 4 10] Naive Total Running Time: 8.349201
 
Running naive.exe with size=8000
Total initialization time is 0.791483
Clocal time is 10.197666
[8000 8000 8000 1 5 4 10] Naive Total Running Time: 12.316928
 
Running naive.exe with size=9000
Total initialization time is 0.979871
Clocal time is 14.311838
[9000 9000 9000 1 5 4 10] Naive Total Running Time: 16.952562
 
Running naive.exe with size=10000
Total initialization time is 1.202356
Clocal time is 19.743526
[10000 10000 10000 1 5 4 10] Naive Total Running Time: 22.997041
 
----- NOW RUNNING VARIABLE NODES -----
Running naive.exe with size=12000
Total initialization time is 1.764217
Clocal time is 292.816908
[12000 12000 12000 1 1 4 10] Naive Total Running Time: 296.231100
 
Running naive.exe with size=12000
Total initialization time is 1.756043
Clocal time is 73.648957
[12000 12000 12000 2 2 4 10] Naive Total Running Time: 77.691825
 
Running naive.exe with size=12000
Total initialization time is 1.958230
Clocal time is 34.880367
[12000 12000 12000 3 3 4 10] Naive Total Running Time: 39.197265
 
