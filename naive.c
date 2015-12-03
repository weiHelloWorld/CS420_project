// CS420 Project: naive.c for MMM
// Created by Rylan Dmello on Nov 19 2015 
//
// Naive Algorithm for distributed MMM (Now includes OpenMP!)
//
// A[m, n] * B[n, p] = C[m, p]
// Assumed processors are in rXc grid. 
// There are nt threads
//
// Usage: mpirun -np 6 ./naive.exe m n p r c mult_mode b nt
// Usage: mpirun -np 6 ./naive.exe 4 6 9 2 3 4 0 1
// Make sure r divides BOTH m AND n. 
// Make sure c divides BOTH n AND p. 
// Make sure b divides m/r AND n AND p/c. 
// Make sure nt divides m/r AND p/c. 

#include "support.h"
// #define DEBUG

int main (int argc, char **argv) {

    int procs, rank, m, n, p, r, c, mult_mode, b, nt; 
    double **A, **B, **C, **D;
    double init_time, final_time, diff_time;
    int log_time = 0;
  
    if(argc !=9 ) {
        fprintf(stderr, "Usage: %s m n p r c mult_mode b nt\n", argv[0]);
        exit(0);
    }

    // m, n = size of input A matrix
    // n, p = size of input B matrix
    // r, c = size of nprocessor matrix
    // mult_mode = sequential multiplication type
    // b = blocksize for loop tiling. Ignored if mult_mode!=3
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    p = atoi(argv[3]);
    r = atoi(argv[4]);
    c = atoi(argv[5]);
    mult_mode = atoi(argv[6]);
    b = atoi(argv[7]);
    nt = atoi(argv[8]);
    
    omp_set_num_threads(nt);

    int required = 2; //MPI_THREAD_FUNNELED
    /* MPI_THREAD_FUNNELED: Only main thread makes MPI calls
     * From testing, it seems like we can go upto MPI_THREAD_SERIALIZED
     * but not MPI_THREAD_MULTIPLE.
     */
    int provided; 


    MPI_Init_thread(&argc, &argv, required, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    //printf("MPI Support is %d\n", provided);
    
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

    if (rank == 0) init_time = get_clock();
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
        
   
#ifdef DEBUG
        double t_mat_create = get_clock();
        printf("Matrix initialization time is %lf\n", t_mat_create-init_time);
 
        D = create_matrix(m, p);
        D = seq_MMM(fullA, fullB, m, n, p);
        double t_seq = get_clock();
        printf("Sequential MMM time is %lf\n", t_seq-t_mat_create);

#endif
        
        /*
        printf("Sequential MMM gives: \n");
        log_matrix(seq_MMM(fullA, fullB, m, n, p), m, p); 
        printf("End of sequential output. \n");
        */

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
        //printf("%d: matrices initialized\n", rank);
    }
    else {
        MPI_Recv(&(A[0][0]), m*n/(r*c), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
        MPI_Recv(&(B[0][0]), n*p/(r*c), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    }
    
    double t_init_send, t_Arow, t_Bcol, t_Clocal, t_C;
    if (rank==0) {
        t_init_send = get_clock();
        printf("Total initialization time is %lf\n", t_init_send-init_time);
    }
    
    //log_matrix(A, m/r, n/c);
    //log_matrix(B, n/r, p/c);

    //MPI_Barrier(MPI_COMM_WORLD); // All submatrices have been distributed

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
    if ((rank==0) && (log_time==1)) {
        t_Arow = get_clock();
        printf("Arow time is %lf\n", t_Arow-t_init_send);
    }

    //printf("%d: Arow matrices built\n", rank);
    //if (rank == 0) log_matrix(Arow, m/r, n);
    //MPI_Barrier(MPI_COMM_WORLD);

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
    if ((rank==0)) {
        t_Bcol = get_clock();
        //printf("Bcol time is %lf\n", t_Bcol-t_Arow);
    }
    //printf("%d: Bcol matrices built\n", rank);
    //MPI_Barrier(MPI_COMM_WORLD);
    //if (rank == 0) log_matrix(Bcol, n, p/c);
    
    // Multiply Arow and Bcol to get Clocal
    // This section will be parallelized with OpenMP 
    // to get best utilization of shared-memory nodes

    double **Clocal;
    double temp=0;
    Clocal = create_matrix(m/r,p/c);
    init_zero(Clocal, m/r, p/c); 

    //multiply_selector(mult_mode, Arow, Bcol, Clocal, m/r, n, p/c, b); 
    multiply_omp_row(mult_mode, Arow, Bcol, Clocal, m/r, n, p/c, b, nt); 
    
    if (rank==0) {
        t_Clocal = get_clock();
        printf("Clocal time is %lf\n", t_Clocal-t_Bcol);
    }
    //if (rank == 0) log_matrix(Clocal, m/r, p/c);
    //printf("%d: Clocal matrices built\n", rank);
    //MPI_Barrier(MPI_COMM_WORLD);

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
        if (log_time==1) {
            t_C = get_clock();
            printf("C send/recv time is %lf\n", t_C-t_Clocal);
        }
        /*
        printf("Parallel MMM output is: \n");
        log_matrix(C, m, p);
        printf("End parallel MMM output. \n");
        */
        final_time = get_clock();
        diff_time = final_time - init_time;
        printf("[%d %d %d %d %d %d %d %d] Naive Total Running Time: %lf\n", m, n, p, r, c, mult_mode, b, nt, diff_time);
#ifdef DEBUG
        compare_matrices(C, D, m, p);
        free_matrix(D);
#endif
        free_matrix(C);
    }
    else {
        MPI_Send(&(Clocal[0][0]), m*p/(r*c), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
    }

    free_matrix(A);
    free_matrix(B); 
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
 *
 * Ask the TA if the matrices should be timed
 * before being distributed
 */

