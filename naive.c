// CS420 Project: naive.c for MMM
// Created by Rylan Dmello on Nov 19 2015 
//
// Naive Algorithm for distributed MMM (does not include OpenMP yet)
//
// A[m, n] * B[n, p] = C[m, p]
// Assumed processors are in rXc grid. 
// Usage: mpirun -np 6 ./naive.exe m n p r c
// Usage: mpirun -np 6 ./naive.exe 4 6 9 2 3
// Make sure r divides BOTH m AND n. 
// Make sure c divides BOTH n AND p. 

#include "support.h"

int main (int argc, char **argv) {

    int procs, rank, m, n, p, r, c; 
    double **A, **B, **C;
   
    // m, n = size of full matrix
    // r, c = size of nprocessor matrix
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    p = atoi(argv[3]);
    r = atoi(argv[4]);
    c = atoi(argv[5]);

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

    //printf("%d: (%d, %d)\n", rank, row, col);

    // Now we have to distribute 1 matrix from rank 0 to all the ranks
    A = create_matrix(m/r, n/c);
    B = create_matrix(n/r, p/c);
    if (rank == 0) {
        // The full matrix is created only on Rank 0

        double **fullA; fullA = create_matrix(m, n);
        double **fullB; fullB = create_matrix(n, p);
        init_spec(fullA, m, n); 
        init_spec(fullB, n, p);
        double **tempA; tempA = create_matrix(m/r, n/c);
        double **tempB; tempB = create_matrix(n/r, p/c);
        
        printf("Sequential MMM gives: \n");
        log_matrix(seq_MMM(fullA, fullB, m, n, p), m, p); 
        printf("End of sequential output. \n");

        // Rank 0 sends the parts of matrix A and B to all the other ranks
        for (int i=0; i<r; i++) {
            for (int j=0; j<c; j++) {
                for (int ii=0; ii<m/r; ii++) {
                    for (int jj=0; jj<n/c; jj++) {
                        tempA[ii][jj] = fullA[(i*m/r)+ii][(j*n/c)+jj];
                        if((i==0)&&(j==0)) {
                            A[ii][jj] = fullA[(i*m/r)+ii][(j*n/c)+jj];
                        }
                    }
                }
                for (int ii=0; ii<n/r; ii++) {
                    for (int jj=0; jj<p/c; jj++) {
                        tempB[ii][jj] = fullB[(i*n/r)+ii][(j*p/c)+jj];
                        if((i==0)&&(j==0)) {
                            B[ii][jj] = fullB[(i*n/r)+ii][(j*p/c)+jj];
                        }
                    }
                }
                if ((i==0)&&(j==0)) continue; 
                int dest_coords[2] = {i, j};
                int dest_rank;
                MPI_Cart_rank(comm2D, dest_coords, &dest_rank);
                MPI_Send(&(tempA[0][0]), m*n/(r*c), MPI_DOUBLE, dest_rank, 0, MPI_COMM_WORLD);
                MPI_Send(&(tempB[0][0]), n*p/(r*c), MPI_DOUBLE, dest_rank, 1, MPI_COMM_WORLD);
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
        MPI_Recv(&(B[0][0]), n*p/(r*c), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    }
    
    //log_matrix(A, m/r, n/c);
    //log_matrix(B, n/r, p/c);

    MPI_Barrier(MPI_COMM_WORLD); // All submatrices have been distributed

    double **Arow, **Bcol; 
    Arow = create_matrix(m/r, n);
    Bcol = create_matrix(n, p/c);

    // Build Arow matrix for each processor
    for (int i=0; i<c; i++) {
        double **tempA; tempA = create_matrix(m/r, n/c);
        if (col == i) {
            for (int ii=0; ii<m/r; ii++)
                for (int jj=0; jj<n/c; jj++)
                    tempA[ii][jj]=A[ii][jj]; //THIS CAN BE OPTIMIZED?
        }
        MPI_Bcast(&(tempA[0][0]), m*n/(r*c), MPI_DOUBLE, i, commR);
        for (int ii=0; ii<m/r; ii++){
            for (int jj=0; jj<n/c; jj++) {
                Arow[ii][(i*n/c)+jj]=tempA[ii][jj];
            }
        }
        free_matrix(tempA);
    }
    //printf("%d: Arow matrices built\n", rank);
    //if (rank == 0) log_matrix(Arow, m/r, n);
    MPI_Barrier(MPI_COMM_WORLD);

    // Build Bcol matrix for each processor
    for (int i=0; i<r; i++) {
        double **tempB; tempB = create_matrix(n/r, p/c);
        if (row == i) {
            for (int ii=0; ii<n/r; ii++)
                for (int jj=0; jj<p/c; jj++)
                    tempB[ii][jj]=B[ii][jj]; //THIS CAN BE OPTIMIZED?
        }
        MPI_Bcast(&(tempB[0][0]), n*p/(r*c), MPI_DOUBLE, i, commC);
        for (int ii=0; ii<n/r; ii++){
            for (int jj=0; jj<p/c; jj++) {
                Bcol[ii+(i*n/r)][jj]=tempB[ii][jj];
            }
        }
        free_matrix(tempB);
    }
    //printf("%d: Bcol matrices built\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    //if (rank == 0) log_matrix(Bcol, n, p/c);
    
    // Multiply Arow and Bcol to get Clocal
    double **Clocal;
    double temp=0;
    Clocal = create_matrix(m/r,p/c);
    for(int i=0; i<m/r; i++){
        for(int j=0; j<p/c; j++) {
            temp = 0;
            for (int k=0; k<n; k++) {
                temp += Arow[i][k]*Bcol[k][j];
            }
            Clocal[i][j] = temp; 
        }
    }
    //if (rank == 0) log_matrix(Clocal, m/r, p/c);
    //printf("%d: Clocal matrices built\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);

    // Send Clocals to root and get C
    if (rank == 0) {
        double **tempC; tempC = create_matrix(m/r,p/c);
        C = create_matrix(m, p);
        int source_rank = 0;
        for (int i=0; i<r; i++) {
            for (int j=0; j<c; j++) {
                if ((i==0) && (j==0)) {
                    for (int ii=0; ii<m/r; ii++) 
                        for (int jj=0; jj<p/c; jj++)
                            C[ii][jj] = Clocal[ii][jj];
                }
                else {
                    int source_coords[2] = {i, j}; 
                    MPI_Cart_rank(comm2D, source_coords, &source_rank);
                    MPI_Recv(&(tempC[0][0]), m*p/(r*c), MPI_DOUBLE, source_rank, source_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (int ii=0; ii<m/r; ii++)
                        for (int jj=0; jj<p/c; jj++)
                            C[ii+(i*m/r)][jj+(j*p/c)] = tempC[ii][jj];
                }
            }
        }
        free_matrix(tempC);
        printf("Parallel MMM output is: \n");
        log_matrix(C, m, p);
        printf("End parallel MMM output. \n");
        free_matrix(C);
    }
    else {
        MPI_Send(&(Clocal[0][0]), m*p/(r*c), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
    }
    //printf("Hello I am %d of %d \n", rank, procs);

    //printf("Is this segfault?\n");
    MPI_Barrier(MPI_COMM_WORLD);

    free_matrix(A);
    free_matrix(B); 
    //free_matrix(C); 
    free_matrix(Arow);
    free_matrix(Bcol);
    free_matrix(Clocal);

    MPI_Finalize();
    return 0;
}

/* Notes:
 * We should try to use a 1D array 
 * and see if there is cache performance
 * improvement over 2D array for storing
 * the matrix. 
 */

