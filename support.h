// CS420 Project: support.h for common functions
//
// Matrix ops: create_matrix(), print_matrix(), log_matrix(), free_matrix()
// Matrix initialization: init_zero(), init_spec(), init_rand()
// MMM operations: seq_MMM(), compare_matrices()
// Timing/profiling: get_clock()

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

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


