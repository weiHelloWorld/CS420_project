// CS420 Project: MPI_cannon.c for MMM
// Created by Wei Chen on Nov 17 2015 
//
// MPI implementation of Cannon's algorithm for MMM

// Assumed processors are in rXr grid. 
//
// Usage: mpirun -np r^2 ./MPI_cannon.exe m n p r mult_mode b nt
// Usage: mpirun -np 9 ./MPI_cannon.exe 12 12 9 3 4 0 1

// Make sure:
// 1. r divides m, n, p
// 2. number of processors are r^2
// 3. b divides m/r AND n AND p/c
// 4. nt divides m/r AND p/c

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>   
#include "support.h"

#define DEBUG


int main (int argc, char** argv) {
    MPI_Status status;
    MPI_Request request[4];
    int my_rank, num_of_procs;
    double init_time, final_time, diff_time;

    if(argc != 8 ) {
        fprintf(stderr, "Usage: %s m n p r mult_mode b nt \n", argv[0]);
        exit(0);
    }

    int m, n, p, r, mult_mode, b, nt;
    // m, n = size of input A matrix
    // n, p = size of input B matrix
    // r = size of nprocessor matrix (num of rows should be equal to num of columns)
    // mult_mode = sequential multiplication type
    // b = blocksize for loop tiling. Ignored if mult_mode!=3
    // nt = number of threads
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    p = atoi(argv[3]);
    r = atoi(argv[4]);
    mult_mode = atoi(argv[5]);
    b = atoi(argv[6]);
    nt = atoi(argv[7]);

    omp_set_num_threads(nt);

    int required = MPI_THREAD_FUNNELED; // MPI funcs are called by master thread
    int provided;

    MPI_Init_thread(&argc, &argv, required, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_procs); 

    int row_num_of_procs; // num of rows should be equal to num of cols

    double **A, **B; // two input matrices, matrices could be non-squared
    double **C; // the result
    double **D; // correct result from sequential version
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
        D = create_matrix(size_of_A[0], size_of_B[1]);
        D = seq_MMM(A, B, size_of_A[0], size_of_A[1], size_of_B[1]); // this is the correct result
    }


    C = create_matrix(size_of_A[0], size_of_B[1]);

    size_of_A_block[0] = size_of_A[0] / row_num_of_procs;
    size_of_A_block[1] = size_of_A[1] / row_num_of_procs;
    size_of_B_block[0] = size_of_B[0] / row_num_of_procs;
    size_of_B_block[1] = size_of_B[1] / row_num_of_procs;

    assert(size_of_A_block[1] == size_of_B_block[0]);

    double **A_block, **B_block, **C_block, **temp_C_block, **temp_block_buffer_A, **temp_block_buffer_B; // these are blocks in each processors
                                                                                        // including buffer blocks for communication
    A_block = create_matrix(size_of_A_block[0], size_of_A_block[1]);
    B_block = create_matrix(size_of_B_block[0], size_of_B_block[1]);
    C_block = create_matrix(size_of_A_block[0], size_of_B_block[1]);
    temp_C_block = create_matrix(size_of_A_block[0], size_of_B_block[1]);

    init_zero(C_block, size_of_A_block[0], size_of_B_block[1]); 

    temp_block_buffer_A = create_matrix(size_of_A_block[0], size_of_A_block[1]);
    temp_block_buffer_B = create_matrix(size_of_B_block[0], size_of_B_block[1]);

    // initialization, partition into blocks and send to destination process
    if (my_rank == 0) init_time = get_clock(); // start timer

    if (my_rank == 0) {
        for (int i = 0; i < row_num_of_procs; i ++) {
            for (int j = 0; j < row_num_of_procs; j ++) {
                if (i == 0 && j == 0) { // copy to itself
                    for (int ii = 0; ii < size_of_A_block[0]; ii ++) {
                        for (int jj = 0; jj < size_of_A_block[1]; jj ++) {
                            A_block[ii][jj] = A[ii][jj];  
                        }
                    }
                    for (int ii = 0; ii < size_of_B_block[0]; ii ++) {
                        for (int jj = 0; jj < size_of_B_block[1]; jj ++) {
                            B_block[ii][jj] = B[ii][jj];  
                        }
                    }
                }
                else { 
                    // first send A block to processor 
                    // (i, (j - i + row_num_of_procs) % row_num_of_procs)
                    for (int ii = i * size_of_A_block[0]; ii < (i+1) * size_of_A_block[0]; ii ++) {
                        for (int jj = j * size_of_A_block[1]; jj < (j+1) * size_of_A_block[1]; jj ++) {
                            // copy to a temp buffer before sending out
                            temp_block_buffer_A[ii - i * size_of_A_block[0]][jj - j * size_of_A_block[1]] = A[ii][jj];
                        }
                    }
                    MPI_Send(&(temp_block_buffer_A[0][0]), 
                        size_of_A_block[0] * size_of_A_block[1], 
                        MPI_DOUBLE, 
                        i * row_num_of_procs + ((j - i + row_num_of_procs) % row_num_of_procs),
                        0, /* tag = 0 is for A_block */
                        MPI_COMM_WORLD
                        );

                    // secondly, send B block to procesor
                    // ((i - j + row_num_of_procs) % row_num_of_procs, j)
                    for (int ii = i * size_of_B_block[0]; ii < (i+1) * size_of_B_block[0]; ii ++) {
                        for (int jj = j * size_of_B_block[1]; jj < (j+1) * size_of_B_block[1]; jj ++) {
                            temp_block_buffer_B[ii - i * size_of_B_block[0]][jj - j * size_of_B_block[1]] = B[ii][jj];
                        }
                    }
                    MPI_Send(temp_block_buffer_B[0], 
                        size_of_B_block[0] * size_of_B_block[1], 
                        MPI_DOUBLE, 
                        ((i - j + row_num_of_procs) % row_num_of_procs) * row_num_of_procs + j,
                        1, /* tag = 1 is for B_block */
                        MPI_COMM_WORLD
                        );
                }
            }
        }       
    }
    else { // receive block from rank 0 and store into A_block, B_block
        MPI_Recv(A_block[0], 
            size_of_A_block[0] * size_of_A_block[1],
            MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, MPI_DOUBLE, &count);
        assert(size_of_A_block[0] * size_of_A_block[1] == count);
        assert(status.MPI_SOURCE == 0);
        assert(status.MPI_TAG == 0);

        MPI_Recv(B_block[0], 
            size_of_B_block[0] * size_of_B_block[1],
            MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
    }
    // initialization ends here
    
    #ifdef DEBUG
        // printf("the A block from rank %d is: \n", my_rank);
        // log_matrix(A_block, 3,3);
    #endif


    // shifting and calculate
    for (int index_of_stages = 0; index_of_stages < row_num_of_procs; index_of_stages ++) { // (row_num_of_procs) is equal to the number of stages
        // do multiplication in each stage
        init_zero(temp_C_block, size_of_A_block[0], size_of_B_block[1]);  // there would be some problems without this initialization

        multiply_omp_row(mult_mode, A_block, B_block, temp_C_block, 
                    size_of_A_block[0], size_of_A_block[1], size_of_B_block[1], 
                    b, nt);
        #ifdef DEBUG
        // printf("temp_C_block from rank %d is: \n", my_rank);
        // log_matrix(temp_C_block, size_of_A_block[0], size_of_B_block[1]);
        #endif
        for (int i = 0; i < size_of_A_block[0]; i ++) {
            for (int k = 0; k < size_of_B_block[1]; k ++) {
                C_block[i][k] += temp_C_block[i][k];
            }
        }
        // do shifting after calculation
        for (int i = 0; i < size_of_A_block[0]; i ++) {
            for (int j = 0; j < size_of_A_block[1]; j ++) {
                temp_block_buffer_A[i][j] = A_block[i][j];  // copy to buffer, ready to send
            }   
        }
        int destination_rank = my_rank % row_num_of_procs == 0 ? my_rank + row_num_of_procs - 1 : my_rank - 1;
        MPI_Isend(temp_block_buffer_A[0], 
            size_of_A_block[0] * size_of_A_block[1], 
            MPI_DOUBLE,
            destination_rank,
            0,
            MPI_COMM_WORLD,
            request
            );

        for (int i = 0; i < size_of_B_block[0]; i ++) {
            for (int j = 0; j < size_of_B_block[1]; j ++) {
                temp_block_buffer_B[i][j] = B_block[i][j];  // copy to buffer, ready to send
            }   
        }
        destination_rank = my_rank < row_num_of_procs ? my_rank + (row_num_of_procs - 1) * row_num_of_procs : my_rank - row_num_of_procs;

        #ifdef DEBUG
        // printf("index_of_stages = %d, my_rank = %d, destination_rank = %d\n", index_of_stages, my_rank, destination_rank);
        #endif
        MPI_Isend(temp_block_buffer_B[0], 
            size_of_B_block[0] * size_of_B_block[1], 
            MPI_DOUBLE,
            destination_rank,
            1,
            MPI_COMM_WORLD,
            request + 1
            );
        int source_rank = (my_rank + 1) % row_num_of_procs == 0 ? my_rank - row_num_of_procs + 1 : my_rank + 1;
        MPI_Irecv(A_block[0],
            size_of_A_block[0] * size_of_A_block[1],
            MPI_DOUBLE,
            source_rank,
            0,
            MPI_COMM_WORLD,
            request + 2
            );
        source_rank = my_rank >= (row_num_of_procs - 1) * row_num_of_procs ? my_rank - (row_num_of_procs - 1) * row_num_of_procs : my_rank + row_num_of_procs;
        MPI_Irecv(B_block[0],
            size_of_B_block[0] * size_of_B_block[1],
            MPI_DOUBLE,
            source_rank,
            1,
            MPI_COMM_WORLD,
            request + 3
            );
        MPI_Waitall(4, request, MPI_STATUS_IGNORE);

        
    }   
    #ifdef DEBUG
        // if (my_rank == 4) {
        //     printf("the C block from rank %d is: \n", my_rank);
        //     log_matrix(C_block, 3,3);
        // }
    #endif


    // gather blocks from processors
    double **temp_for_gather; // this is used to receive gathered results
    temp_for_gather = create_matrix(num_of_procs, size_of_A_block[0] * size_of_B_block[1]);

    MPI_Gather(C_block[0], size_of_A_block[0] * size_of_B_block[1], MPI_DOUBLE,
               temp_for_gather[0], size_of_A_block[0] * size_of_B_block[1], MPI_DOUBLE,
               0, MPI_COMM_WORLD
              );

    // copy to C as result
    for (int i = 0; i < num_of_procs; i ++) {
        int start_of_row_index, start_of_col_index;
        start_of_row_index = (i / row_num_of_procs) * size_of_A_block[0];
        start_of_col_index = (i % row_num_of_procs) * size_of_B_block[1];
        for (int j = 0; j < size_of_A_block[0]; j ++) {
            for (int k = 0; k < size_of_B_block[1]; k ++) {
                C[start_of_row_index + j][start_of_col_index + k] = temp_for_gather[i][j * size_of_B_block[1] + k];
            }
        }
    }

    if (my_rank == 0) {
        final_time = get_clock();
        diff_time = final_time - init_time;
        printf("[%d %d %d %d %d %d %d] MPI_cannon Total Running Time: %lf\n", m, n, p, r, mult_mode, b, nt, diff_time);
    }

    #ifdef DEBUG
    if (my_rank == 0) {
            compare_matrices(C, D, size_of_A[0], size_of_B[1]);
            printf("C from rank %d is: \n", my_rank);
            log_matrix(C, size_of_A[0], size_of_B[1]);
        }
    #endif
    


    free_matrix(A_block);
    free_matrix(B_block);
    
    free_matrix(temp_for_gather);
    free_matrix(temp_block_buffer_A);
    free_matrix(temp_block_buffer_B);

    if (my_rank == 0) {
        free_matrix(A);
        free_matrix(B);
        free_matrix(C);
        free_matrix(D);
    }

    MPI_Finalize();
    return 0;
}

