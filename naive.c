// CS420 Project: naive.c for MMM
// Created by Rylan Dmello on Nov 19 2015 
// Naive Algorithm for distributed MMM (does not include OpenMP yet)
// A[m, n] * B[n, m] = C[m, m]
// Usage: mpirun -np 2 ./naive.exe 10 10

#include "support.h"

// Main
int main (int argc, char **argv) {

    int procs, rank, m, n, r, c; 
    double **A, **B, **C;
   
    // m, n = size of actual matrix
    // r, c = size of nprocessor matrix
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    r = atoi(argv[3]);
    c = atoi(argv[4]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    
    // The 2D Cart_create and split algorithm is from the
    // MPI Calls lecture on Fri, Oct 9. See the wiki for details
    int dimSize[2] = {r, c};
    int periodic[2] = {0, 0};
    int cart_coords[2];
    MPI_Comm comm2D, commR, commC;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dimSize, periodic, 0, &comm2D);

    int row, col; 
    MPI_Cart_coords(comm2D, rank, 2, cart_coords);
    row = cart_coords[0]; col = cart_coords[1];
    
    MPI_Comm_split(comm2D, row, col, &commR);  // Row communicator
    MPI_Comm_split(comm2D, col, row, &commC);  // Column communicator

    // Now we have to distribute 1 matrix from rank 0 to all the ranks
    if (rank == 0) {
        fullA = create_matrix(m, n);
        fullB = create_matrix(n, m);




    }


    printf("Hello I am %d of %d \n", rank, procs);

    MPI_Finalize();

    A = create_matrix(m, n);
    init_spec(A, m, n);
    log_matrix(A, m, n);
    free_matrix(A);

    return 0;
}

/* Notes:
 * We should try to use a 1D array 
 * and see if there is cache performance
 * improvement over 2D array for storing
 * the matrix. 
 */

