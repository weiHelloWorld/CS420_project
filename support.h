// CS420 Project: support.h for common functions
//
// Matrix ops: create_matrix(), print_matrix(), log_matrix(), free_matrix()
// Matrix initialization: init_zero(), init_spec(), init_rand()
// MMM operations: seq_MMM(), compare_matrices()
// MMM methods: multiply_selector(), multiply_basic, multiply_urjam, multiply_tiled, multiply_BLAS
// OpenMP methods: 
// Timing/profiling: get_clock()

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl.h>
#include <omp.h>
#include <assert.h>   

// #define DEBUG


// create_matrix function is the exact same one
// used for mp1 in mmm_basic.c
double** create_matrix(int m, int n) {
    double **mat; 
    int i;

    mat = (double**) malloc(m*sizeof(double*));
    mat[0] = (double*)malloc(m*n*sizeof(double));

    for (i=1; i<m; i++) 
        mat[i] = mat[0] + i*n;
    
    return mat;
}

// print_matrix fcn is also from mmm_basic.c in mp1
void print_matrix (double **mat, int m, int n) {
    FILE* fp;
    int i, j;

    fp = fopen("verify.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "Unable to write the file 'verify.txt'\n");
        exit(1);
    }

    for (i=0; i<m; i++) {
        for (j=0; j<n; j++) {
            fprintf(fp, "%lf\t", mat[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

// print_matrix to stdout
void log_matrix (double **mat, int m, int n) {
    int i, j;
    for (i=0; i<m; i++){
        for (j=0; j<n; j++){
            printf("%lf\t", mat[i][j]);
        }
        printf("\n");
    }
}

//free_matrix from mmm_basic.c in mp1
void free_matrix(double **M){
    free(M[0]);
    free(M);
}

// Initialize matrices to zero matrix
// From mmm_basic.c in MP1
void init_zero(double **mat, int m, int n) {
    int i, j;
    for (i=0; i<m; i++)
        for (j=0; j<n; j++)
            mat[i][j]=0;
}

// Initialize matrices to specific integer values
// from mmm_basic in MP1
void init_spec(double **mat, int m, int n) {
    int i, j;
    for (i=0; i<m; i++)
        for (j=0; j<n; j++)
            mat[i][j]=2*i+j;
}

// Initialize matrices to random values
// from mmm_basic in MP1
void init_rand(double **mat, int m, int n) {
    int i, j;
    for (i=0; i<m; i++)
        for (j=0; j<n; j++)
            mat[i][j]=rand();
}

// seq_MMM multiplies an m*n matrix by a n*p matrix and
// returns a m*p matrix. 
double** seq_MMM (double **A, double **B, int m, int n, int p) {
    double **C; C = create_matrix(m, p); 
    init_zero(C, m, p);
    for (int i=0; i<m; i++) 
        for (int j=0; j<p; j++) 
            for (int k=0; k<n; k++) 
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

// compare_matrices checks if two matrices C and D of aXb dimensions are equal
// and prints the unequal parts of the matrix if unequal
void compare_matrices (double **C, double **D, int a, int b) {
    double **E; E=create_matrix(a,b); init_zero(E, a, b);
    int count = 0;
    double epsilon = 0.000000001;
    //double epsilon = 0.01;//different value for comparison

    for (int i=0; i<a; i++){
        for (int j=0; j<b; j++) {
            if(abs(C[i][j]-D[i][j])<epsilon) {
                E[i][j]=1;
                count++;
            }
        }
    }
    if (count == a*b) printf("Matrices are equal\n");
    else {
        printf("Matrices are unequal.\n");
        printf("%d out of %d values match\n", count, a*b);
        printf("Matrix of values (1=matching, 0=not matching)\n");
        log_matrix(E, a, b);
    }
}

// multiply_basic is similar to seq_MMM above, however, 
// it uses a predefined input matrix. 
// This can be used to understand the speed differences between
// loop tiling, unrolling, and unoptimized. 
// This is similar to MP1 mmm_basic.c
void multiply_basic(double **A, double **B, double **C, int m, int n, int p) {
    #ifdef DEBUG
    // log_matrix(A, m, n);
    printf("m = %d, n = %d, p = %d\n", m, n, p);
    #endif

    for (int i=0; i<m; i++) 
        for (int j=0; j<p; j++)
            for (int k=0; k<n; k++)
                C[i][j] += A[i][k] * B[k][j];
}

// multiply_basic_opt applies a basic optimization
// compiler should be doing this anyway
// but it's good to double check
void multiply_basic_opt(double **A, double **B, double **C, int m, int n, int p) {
    double temp; 
    
    for (int i=0; i<m; i++) {
        for (int j=0; j<p; j++) {
            temp = 0;  
            for (int k=0; k<n; k++) {
                temp += A[i][k] * B[k][j];
            }
            C[i][j] = temp;
        }
    }
}

// multiply_urjam_2 applies 2-way loop unrolling in two directions
// we could make this varible-way loop unrolling too
void multiply_urjam_2(double **A, double **B, double **C, int m, int n, int p) {
    assert(m % 2 == 0);
    assert(n % 2 == 0);
    #ifdef DEBUG
    // printf("in urjam: A is: \n");
    // log_matrix(A, m, n);
    printf("m = %d, n = %d, p = %d\n", m, n, p);
    #endif
    for (int i=0; i<m; i=i+2) {
        for (int j=0; j<p; j=j+2) {
            for (int k=0; k<n; k++) {
                C[i][j] += A[i][k] * B[k][j];
                C[i+1][j] += A[i+1][k] * B[k][j];
                C[i][j+1] += A[i][k] * B[k][j+1];
                C[i+1][j+1] += A[i+1][k] * B[k][j+1];
            }
        }
    }
}

// multiply_tiled applies loop tiling
// note the extra input parameter b for blocksize
void multiply_tiled(double **A, double **B, double **C, int m, int n, int p, int b) {
    int ii,jj,kk,i, j, k;
    double temp; 
    
    for(ii=0; ii<m; ii=ii+b)
        for(jj=0; jj<p; jj=jj+b)
            for(kk=0; kk<n; kk=kk+b)
                for(i = 0; i < b; i++)
                    for(j = 0; j < b; j++) {
                        for(k = 0; k < b; k++)
                           C[ii+i][jj+j] += A[ii+i][kk+k] * B[kk+k][jj+j];
                    }
}

// multiply_BLAS uses the Intel MKL vendor supplied BLAS subroutines
// to multiply two dense matrices
// See: https://software.intel.com/en-us/node/429920
void multiply_BLAS(double **A, double **B, double **C, int m, int n, int p) {
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            m, p, n, 1, &(A[0][0]), n, &(B[0][0]), p, 0, &(C[0][0]), p); 
}

// multiply_selector takes an integer "Multiply Mode" as input
// and uses the corresponding sequential multiplcation algorithm
// 0: basic unoptimzied
// 1: basic optimized
// 2: unroll-and-jam
// 3: loop tiling
// 4: BLAS/MKL highly optimized MMM
void multiply_selector(int mult_mode,
                       double **A, double **B, double **C, 
                       int m, int n, int p,
                       int b) {
    if (mult_mode == 0)
        multiply_basic(A, B, C, m, n, p);
    else if (mult_mode == 1) 
        multiply_basic_opt(A, B, C, m, n, p); 
    else if (mult_mode == 2) 
        multiply_urjam_2(A, B, C, m, n, p); 
    else if (mult_mode == 3) 
        multiply_tiled(A, B, C, m, n, p, b); 
    else if (mult_mode == 4) 
        multiply_BLAS(A, B, C, m, n, p); 
    else {
        fprintf(stderr, "Multiply_selector: Incorrect value for mult_mode. Use 0,1,2,3,4.\n");
        exit(0);
    }   
}

// Multiply_omp_row splits a single MMM into multiple threads
// this is row decomposition. 
void multiply_omp_row (int mult_mode, 
                       double **A, double **B, double **C, 
                       int m, int n, int p, 
                       int b, 
                       int nt) {
    
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        int nt2 = omp_get_num_threads();
        int mystart = t*m/nt2;
        
        multiply_selector(mult_mode, &(A[mystart]), B, &(C[mystart]),m/nt2,n,p,b);
        //printf("%d: Number of threads is %d\n", t, nt2);
    }
}

// get_clock() from support.h in MP1
double get_clock() {
    struct timeval tv; 
    int ok;
    ok = gettimeofday(&tv, NULL);
    if (ok < 0) {
        fprintf(stderr, "Timer gettimeofday error!\n");
        exit(1);
    }
    return tv.tv_sec*1.0 + tv.tv_usec*1.0e-6;
}


