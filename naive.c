// CS420 Project: naive.c for MMM
// Created by Rylan Dmello on Nov 19 2015 
// Naive Algorithm for distributed MMM (does not include OpenMP yet)

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

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

//free_matrix from mmm_basic.c in mp1
void free_matrix(double **M){
    free(M[0]);
    free(M);
}

// Main
int main (int argc, char **argv) {

    int procs, rank, m, n; 
    double **A;

    m = atoi(argv[1]);
    n = atoi(argv[2]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    printf("Hello I am %d of %d \n", rank, procs);

    MPI_Finalize();

    A = create_matrix(m, n);
    print_matrix(A, m, n);
    free_matrix(A);

    return 0;
}

/* Notes:
 * We should try to use a 1D array 
 * and see if there is cache performance
 * improvement over 2D array for storing
 * the matrix. 
 */
