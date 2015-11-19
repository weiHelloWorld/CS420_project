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
   
    // m, n = size of full matrix
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

    printf("%d: (%d, %d)\n", rank, row, col);

    // Now we have to distribute 1 matrix from rank 0 to all the ranks
    A = create_matrix(m/r, n/c);
    B = create_matrix(n/c, m/r);
    if (rank == 0) {
        // The full matrix is created only on Rank 0

        double **fullA; fullA = create_matrix(m, n);
        double **fullB; fullB = create_matrix(n, m);
        init_spec(fullA, m, n); 
        init_spec(fullB, n, m);
        double **tempA; tempA = create_matrix(m/r, n/c);
        double **tempB; tempB = create_matrix(n/c, m/r);

        // Rank 0 sends the parts of matrix A to all the other ranks
        for (int i=0; i<r; i++) {
            for (int j=0; j<c; j++) {
                for (int ii=0; ii<m/r; ii++) {
                    for (int jj=0; jj<n/c; jj++) {
                        tempA[ii][jj] = fullA[(i*m/r)+ii][(j*n/c)+jj];
                        tempB[jj][ii] = fullB[(j*n/c)+jj][(i*m/r)+ii];
                        if((i==0)&&(j==0)) {
                            A[ii][jj] = fullA[(i*m/r)+ii][(j*n/c)+jj];
                            B[jj][ii] = fullB[(j*n/c)+jj][(i*m/r)+ii];
                        }
                    }
                }
                if ((i==0)&&(j==0)) continue; 
                int dest_coords[2] = {i, j};
                int dest_rank;
                MPI_Cart_rank(comm2D, dest_coords, &dest_rank);
                MPI_Send(&(tempA[0][0]), m*n/(r*c), MPI_DOUBLE, dest_rank, 0, MPI_COMM_WORLD);
                MPI_Send(&(tempB[0][0]), m*n/(r*c), MPI_DOUBLE, dest_rank, 1, MPI_COMM_WORLD);
            }
        }
        free_matrix(fullA); 
        free_matrix(fullB);
        free_matrix(tempA);
        free_matrix(tempB);
        printf("%d: matrices initialized\n", rank);
    }
    else {
        MPI_Recv(&(A[0][0]), m*n/(r*c), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
        MPI_Recv(&(B[0][0]), m*n/(r*c), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    }
    
    //log_matrix(A, m/r, n/c);
    //log_matrix(B, n/c, m/r);

    MPI_Barrier(MPI_COMM_WORLD); // All submatrices have been distributed

    double **Arow, **Bcol; 
    Arow = create_matrix(m/r, n);
    Bcol = create_matrix(m, n/c);

    //MPI_Allgather(&(A[0][0]), m*n/(r*c), MPI_DOUBLE, &(Arow[0][0]), m*n/r, MPI_DOUBLE, commR);
    //Send matrices, build subtotals, add, and send back to rank 0

    printf("Hello I am %d of %d \n", rank, procs);

    free_matrix(A);
    free_matrix(B); 
    free_matrix(Arow);
    free_matrix(Bcol);

    MPI_Finalize();
    return 0;
}

/* Notes:
 * We should try to use a 1D array 
 * and see if there is cache performance
 * improvement over 2D array for storing
 * the matrix. 
 */

