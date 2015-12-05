// CS420 Project: MPI_OpenMP_cannon.c for MMM
// Created by Wei Chen on Nov 17 2015 
//
// hybrid implementation of Cannon's algorithm for MMM

// Assumed processors are in rXr grid. 
//
// Usage: mpirun -np r ./MPI_OpenMP_cannon.exe m n p r 
// Usage: mpirun -np 3 ./MPI_OpenMP_cannon.exe 12 12 9 3

// Make sure:
// 1. r divides m, n, p
// 2. number of processors are r

// divide matrices A and B into n^2 blocks, the blocks in the same row are in the same processor, 
// but in different threads, the blocks in different rows are in different processors
// inputs: matrices A and B, number of row (which is the number of processors and also the number of 
// threads in each processor)


#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>   
#include "support.h"

// #define DEBUG

int main(int argc, char* argv[]) {
    MPI_Status status;
    MPI_Request request[2];

    int my_rank, num_of_procs;
    double init_time, final_time, total_time, comp_time = 0, comp_start, comp_end;

    if(argc != 5) {
        fprintf(stderr, "Usage: %s m n p r\n", argv[0]);
        exit(0);
    }

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
    }    
    if (my_rank == 0) {
        C = create_matrix(size_of_A[0], size_of_B[1]);
    }
    else {
        C = create_matrix(1,1); // this resolves seg fault, no idea why
    }

    size_of_A_block[0] = size_of_A[0] / row_num_of_procs;
    size_of_A_block[1] = size_of_A[1] / row_num_of_procs;
    size_of_B_block[0] = size_of_B[0] / row_num_of_procs;
    size_of_B_block[1] = size_of_B[1] / row_num_of_procs;

    assert(size_of_A_block[1] == size_of_B_block[0]);

    double **A_row, **B_row, **C_row, **temp_buffer_A_row, **temp_buffer_B_row; 
    double **A_block, **B_block, **C_block;

    A_row = create_matrix(size_of_A_block[0], size_of_A[1]);
    B_row = create_matrix(size_of_B_block[0], size_of_B[1]);
    C_row = create_matrix(size_of_A_block[0], size_of_B[1]);

    init_zero(C_row, size_of_A_block[0], size_of_B[1]);

    temp_buffer_A_row = create_matrix(size_of_A_block[0], size_of_A[1]);
    temp_buffer_B_row = create_matrix(size_of_B_block[0], size_of_B[1]);

    init_time = get_clock();

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
                MPI_Isend(temp_buffer_A_row[0], 
                        size_of_A_block[0] * size_of_A[1], 
                        MPI_DOUBLE, 
                        i,
                        0, /* tag = 0 is for A_row */
                        MPI_COMM_WORLD,
                        request
                        );
                for (int j = 0; j < row_num_of_procs; j ++) {
                    for (int ii = 0; ii < size_of_B_block[0]; ii ++) {
                        for (int jj = 0; jj < size_of_B_block[1]; jj ++) {

                            temp_buffer_B_row[ii][jj + j * size_of_B_block[1]] 
                                = B[ii + ((i + j) % row_num_of_procs) * size_of_B_block[0]][jj + j * size_of_B_block[1]];

                        }
                    }
                }
                MPI_Isend(temp_buffer_B_row[0], 
                        size_of_B_block[0] * size_of_B[1], 
                        MPI_DOUBLE, 
                        i, 
                        1, /* tag = 1 is for B_row */
                        MPI_COMM_WORLD,
                        request + 1
                        );
                MPI_Waitall(2, request, MPI_STATUS_IGNORE);
            }
        }
    }
    else {
        MPI_Irecv(A_row[0], 
            size_of_A_block[0] * size_of_A[1],
            MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, request);
        MPI_Irecv(B_row[0], 
            size_of_B_block[0] * size_of_B[1],
            MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, request + 1);
        MPI_Waitall(2, request, MPI_STATUS_IGNORE);
    }



    // multiply and shifting
    omp_set_num_threads(row_num_of_procs);
    for (int iteration = 0; iteration < row_num_of_procs; iteration ++) {
        // multiply
        comp_start = get_clock();

        #pragma omp parallel private (A_block, B_block, C_block)
        {
            int my_thread_index;
            my_thread_index = omp_get_thread_num();
            
            A_block = create_matrix(size_of_A_block[0], size_of_A_block[1]);
            B_block = create_matrix(size_of_B_block[0], size_of_B_block[1]);
            C_block = create_matrix(size_of_A_block[0], size_of_B_block[1]);

            // copy corresponding part in A_row and B_row into A_block, B_block, for calling computation function
            for (int i = 0; i < size_of_A_block[0]; i ++) {
                for (int j = 0; j < size_of_A_block[1]; j ++) {
                    A_block[i][j] = A_row[i][( (my_rank + my_thread_index + iteration) % row_num_of_procs) * size_of_A_block[1] + j];
                    /* here we do shifting in A by adding an offset to column index */
                }
            }

            for (int i = 0; i < size_of_B_block[0]; i ++) {
                for (int j = 0; j < size_of_B_block[1]; j ++) {
                    B_block[i][j] = B_row[i][my_thread_index * size_of_B_block[1] + j];
                }
            }
            multiply_BLAS(A_block, B_block, C_block, size_of_A_block[0], size_of_A_block[1], size_of_B_block[1]);

            free_matrix(A_block);
            free_matrix(B_block);

            for (int i = 0; i < size_of_A_block[0]; i ++) {
                for (int j = 0; j < size_of_B_block[1]; j ++) {
                    C_row[i][my_thread_index * size_of_B_block[1] + j] += C_block[i][j];
                }
            }   


            free_matrix(C_block);

            // for (int i = 0; i < size_of_A_block[0]; i ++) {
            //     for (int j = 0; j < size_of_B_block[1]; j ++) {
            //         for (int k = 0; k < size_of_A_block[1]; k ++) {
            //             C_row[i][my_thread_index * size_of_B_block[1] + j] 
            //                 += A_row[i][( (my_rank + my_thread_index + iteration) % row_num_of_procs) * size_of_A_block[1] + k]
            //                 * B_row[k][my_thread_index * size_of_B_block[1] + j];    
            //                 /* here we do shifting in A by adding an offset to column index */
            //         }
            //     }
            // }   
        }
        comp_end = get_clock();
        comp_time += (comp_end - comp_start);
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
    }

    // gather

    MPI_Gather(C_row[0], size_of_A_block[0] * size_of_B[1], MPI_DOUBLE,
               C[0], size_of_A_block[0] * size_of_B[1], MPI_DOUBLE,
               0, MPI_COMM_WORLD
              );

    if (my_rank == 0) {
        final_time = get_clock();
        total_time = final_time - init_time;
        printf("[%d %d %d %d %d %d] MPI_OpenMP_cannon Total Running Time: %lf\n", m, n, p, r, mult_mode, b, total_time);
        printf("[%d %d %d %d %d %d] MPI_OpenMP_cannon Total Computation Time: %lf\n", m, n, p, r, mult_mode, b, comp_time);        
    }
    

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
        #ifdef DEBUG
        free_matrix(D);
        #endif
    }
    free_matrix(C);

    MPI_Finalize();
    return 0;
}

