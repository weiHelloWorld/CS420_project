// divide matrices A and B into n^2 blocks, the blocks in the same row are in the same processor, 
// but in different threads, the blocks in different rows are in different processors
// inputs: matrices A and B, number of row (which is the number of processors and also the number of 
// threads in each processor)

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>   
#include "support.h"

#define DEBUG

int main(int argc, char* argv[]) {
    MPI_Status status;
    MPI_Request request[2];

    int my_rank, num_of_procs;

    int m, n, p, r, mult_mode, b;
    // m, n = size of input A matrix
    // n, p = size of input B matrix
    // r = size of nprocessor matrix (num of rows should be equal to num of columns)
    // mult_mode = sequential multiplication type
    // b = blocksize for loop tiling. Ignored if mult_mode!=3
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    p = atoi(argv[3]);
    r = atoi(argv[4]);
    mult_mode = atoi(argv[5]);
    b = atoi(argv[6]);

    int required = MPI_THREAD_FUNNELED; // MPI funcs are called by master thread
    int provided;

    MPI_Init_thread(&argc, &argv, required, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_procs); 

    int row_num_of_procs; // num of rows should be equal to num of cols

    double **A, **B; // two input matrices, matrices could be non-squared
    double **C; // the result
    double **D;
    int size_of_A[2], size_of_B[2];
    int size_of_A_block[2], size_of_B_block[2];

    // for rank 0: get the input of the two matrices 

    size_of_A[0] = m; size_of_A[1] = n;
    size_of_B[0] = n; size_of_B[1] = p;
    row_num_of_procs = r;

    if (my_rank == 0) {
        A = create_matrix(size_of_A[0], size_of_A[1]);
        B = create_matrix(size_of_B[0], size_of_B[1]);
        init_spec(A, size_of_A[0], size_of_A[1]);
        init_spec(B, size_of_B[0], size_of_B[1]);
        #ifdef DEBUG
        D = create_matrix(size_of_A[0], size_of_B[1]);
        D = seq_MMM(A, B, size_of_A[0], size_of_A[1], size_of_B[1]); // this is the correct result
        #endif
        C = create_matrix(size_of_A[0], size_of_B[1]);
    }    

    size_of_A_block[0] = size_of_A[0] / row_num_of_procs;
    size_of_A_block[1] = size_of_A[1] / row_num_of_procs;
    size_of_B_block[0] = size_of_B[0] / row_num_of_procs;
    size_of_B_block[1] = size_of_B[1] / row_num_of_procs;

    assert(size_of_A_block[1] == size_of_B_block[0]);

    double **A_row, **B_row, **C_row, **temp_buffer_A_row, **temp_buffer_B_row; 

    A_row = create_matrix(size_of_A_block[0], size_of_A[1]);
    B_row = create_matrix(size_of_B_block[0], size_of_B[1]);
    C_row = create_matrix(size_of_A_block[0], size_of_B[1]);

    init_zero(C_row, size_of_A_block[0], size_of_B[1]);

    temp_buffer_A_row = create_matrix(size_of_A_block[0], size_of_A[1]);
    temp_buffer_B_row = create_matrix(size_of_B_block[0], size_of_B[1]);

    // initialization: send blocks to specific row in specific processor
    // for A, simply copy the whole row of blocks into specific processor
    // for B, we need to do "initialization shifting" manually before sending the row
    if (my_rank == 0) {
        for (int i = 0; i < row_num_of_procs; i ++) {
            if (i == 0) {
                for (int ii = 0; ii < size_of_A_block[0]; ii ++) {
                    for (int jj = 0; jj < size_of_A[1]; jj ++) {
                        A_row[ii][jj] = A[ii][jj];
                    }
                }
                for (int j = 0; j < row_num_of_procs; j ++) {
                    // this additional loop var is used to do initialization shifting
                    for (int ii = 0; ii < size_of_B_block[0]; ii ++) {
                        for (int jj = 0; jj < size_of_B_block[1]; jj ++) {
                            B_row[ii][jj + j * size_of_B_block[1]] = B[ii + j * size_of_B_block[0]][jj + j * size_of_B_block[1]];
                        }
                    }
                }
            }
            else {
                for (int ii = i * size_of_A_block[0]; ii < (i + 1) * size_of_A_block[0]; ii ++) {
                    for (int jj = 0; jj < size_of_A[1]; jj ++) {
                        temp_buffer_A_row[ii - i * size_of_A_block[0]][jj] = A[ii][jj];
                    }
                }
                MPI_Send(temp_buffer_A_row[0], 
                        size_of_A_block[0] * size_of_A[1], 
                        MPI_DOUBLE, 
                        i,
                        0, /* tag = 0 is for A_row */
                        MPI_COMM_WORLD
                        );
                for (int j = 0; j < row_num_of_procs; j ++) {
                    for (int ii = 0; ii < size_of_B_block[0]; ii ++) {
                        for (int jj = 0; jj < size_of_B_block[1]; jj ++) {
                            #ifdef DEBUG
                            // printf("temp_buffer_B_row[%d][%d] = B[%d][%d] \n", 
                            //         ii, jj + j * size_of_B_block[1],
                            //         ii + ((i + j) % row_num_of_procs) * size_of_B_block[0], 
                            //         jj + j * size_of_B_block[1]);
                            #endif

                            temp_buffer_B_row[ii][jj + j * size_of_B_block[1]] 
                                = B[ii + ((i + j) % row_num_of_procs) * size_of_B_block[0]][jj + j * size_of_B_block[1]];

                        }
                    }
                }
                MPI_Send(temp_buffer_B_row[0], 
                        size_of_B_block[0] * size_of_B[1], 
                        MPI_DOUBLE, 
                        i, 
                        1, /* tag = 1 is for B_row */
                        MPI_COMM_WORLD
                        );
            }
        }
    }
    else {
        MPI_Recv(A_row[0], 
            size_of_A_block[0] * size_of_A[1],
            MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(B_row[0], 
            size_of_B_block[0] * size_of_B[1],
            MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
    }

#ifdef DEBUG
    // if (my_rank == 2) {
    //     printf("the A row from rank %d is: \n", my_rank);
    //     log_matrix(A_row, 3,9);
    //     printf("the B row from rank %d is: \n", my_rank);
    //     log_matrix(B_row, 3,9);
    // }
#endif

    // multiply and shifting
    omp_set_num_threads(row_num_of_procs);
    for (int iteration = 0; iteration < row_num_of_procs; iteration ++) {
        // multiply
        #pragma omp parallel 
        {
            int my_thread_index;
            my_thread_index = omp_get_thread_num();
            for (int i = 0; i < size_of_A_block[0]; i ++) {
                for (int j = 0; j < size_of_A_block[1]; j ++) {
                    for (int k = 0; k < size_of_A_block[1]; k ++) {
                        C_row[i][my_thread_index * size_of_B_block[1] + j] 
                            += A_row[i][( (my_rank + my_thread_index + iteration) % row_num_of_procs) * size_of_A_block[1] + k]
                            * B_row[k][my_thread_index * size_of_B_block[1] + j];    
                            /* here we do shifting in A by adding an offset to column index */
                    }
                }
            }
        }
        // shifting B
        for (int i = 0; i < size_of_B_block[0]; i ++) {
            for (int j = 0; j < size_of_B[1]; j ++) {
                temp_buffer_B_row[i][j] = B_row[i][j];
            }
        }

        MPI_Isend(temp_buffer_B_row[0], 
                size_of_B_block[0] * size_of_B[1], 
                MPI_DOUBLE, 
                (my_rank + row_num_of_procs - 1) % row_num_of_procs,
                1, 
                MPI_COMM_WORLD,
                request
                );
        MPI_Irecv(B_row[0],
            size_of_B_block[0] * size_of_B[1], 
            MPI_DOUBLE,
            (my_rank + 1) % row_num_of_procs,
            1,
            MPI_COMM_WORLD,
            request + 1
            );
        MPI_Waitall(2, request, MPI_STATUS_IGNORE);
        #ifdef DEBUG
        // if (my_rank == 0) {
        //         printf("iteration %d: the B_row from rank %d is: \n", iteration, my_rank);
        //         log_matrix(B_row, 3,9);
        //     }
        #endif
    }

    // gather

    MPI_Gather(C_row[0], size_of_A_block[0] * size_of_B[1], MPI_DOUBLE,
               C[0], size_of_A_block[0] * size_of_B[1], MPI_DOUBLE,
               0, MPI_COMM_WORLD
              );

    #ifdef DEBUG
    if (my_rank == 0) {
        compare_matrices(C, D, size_of_A[0], size_of_B[1]);
    }
    #endif

    free_matrix(A_row);
    free_matrix(B_row);
    free_matrix(C_row);

    if (my_rank == 0) {
        free_matrix(A);
        free_matrix(B);
        free_matrix(C);
        #ifdef DEBUG
        free_matrix(D);
        #endif
    }

    MPI_Finalize();
    return 0;
}

