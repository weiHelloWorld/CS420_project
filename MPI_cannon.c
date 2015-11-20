#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>   
#include "support.h"

#define DEBUG

int main(int argc, char* argv[]) {
    MPI_Status status;
    int my_rank, num_of_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_procs); 

    int num_of_rows_and_cols_of_procs; // num of rows should be equal to num of cols

    double **A, **B; // two input matrices, matrices could be non-squared
    int size_of_matrix_A[2], size_of_matrix_B[2];
    int size_of_A_block[2], size_of_B_block[2];

    #ifdef DEBUG
    A = create_matrix(9, 9);
    B = create_matrix(9, 9);
    init_spec(A, 9, 9);
    init_spec(B, 9, 9);
    size_of_matrix_A[0] = 9; size_of_matrix_A[1] = 9;
    size_of_matrix_B[0] = 9; size_of_matrix_B[1] = 9;
    num_of_rows_and_cols_of_procs = 3;
    if (my_rank == 0) {
        print_matrix(A, 9, 9);
    }

    #endif

    size_of_A_block[0] = size_of_matrix_A[0] / num_of_rows_and_cols_of_procs;
    size_of_A_block[1] = size_of_matrix_A[1] / num_of_rows_and_cols_of_procs;
    size_of_B_block[0] = size_of_matrix_B[0] / num_of_rows_and_cols_of_procs;
    size_of_B_block[1] = size_of_matrix_B[1] / num_of_rows_and_cols_of_procs;

    double **A_block, **B_block, **C_block, **temp_block_buffer_A, **temp_block_buffer_B; 
    A_block = create_matrix(size_of_A_block[0], size_of_A_block[1]);
    B_block = create_matrix(size_of_B_block[0], size_of_B_block[1]);

    temp_block_buffer_A = create_matrix(size_of_A_block[0], size_of_A_block[1]);
    temp_block_buffer_B = create_matrix(size_of_B_block[0], size_of_B_block[1]);


    // for rank 0: get the input of the two matrices (maybe from files)



    // initialization, partition into blocks and send to destination process
    if (my_rank == 0) {
        for (int i = 0; i < num_of_rows_and_cols_of_procs; i ++) {
            for (int j = 0; j < num_of_rows_and_cols_of_procs; j ++) {
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
                    // (i, (j - i + num_of_rows_and_cols_of_procs) % num_of_rows_and_cols_of_procs)
                    for (int ii = i * size_of_A_block[0]; ii < (i+1) * size_of_A_block[0]; ii ++) {
                        for (int jj = j * size_of_A_block[1]; jj < (j+1) * size_of_A_block[1]; jj ++) {
                            // copy to a temp buffer before sending out
                            temp_block_buffer_A[ii - i * size_of_A_block[0]][jj - j * size_of_A_block[1]] = A[ii][jj];
                        }
                    }
                    MPI_Send(&(temp_block_buffer_A[0][0]), 
                        size_of_A_block[0] * size_of_A_block[1], 
                        MPI_DOUBLE, 
                        i * num_of_rows_and_cols_of_procs + ((j - i + num_of_rows_and_cols_of_procs) % num_of_rows_and_cols_of_procs),
                        0, /* tag = 0 is for A_block */
                        MPI_COMM_WORLD
                        );
                    #ifdef DEBUG
                    if (i == 0 && j == 1) {
                        printf("send to processor %d\n", i * num_of_rows_and_cols_of_procs + ((j - i + num_of_rows_and_cols_of_procs) % num_of_rows_and_cols_of_procs));
                        printf("temp_block_buffer_A = \n");
                        log_matrix(temp_block_buffer_A, 3, 3);
                    }
                    assert(num_of_procs == 9);
                    #endif

                    assert(size_of_A_block[0] * size_of_A_block[1] == num_of_procs);
                    // secondly, send B block to procesor
                    // ((i - j + num_of_rows_and_cols_of_procs) % num_of_rows_and_cols_of_procs, j)
                    for (int ii = i * size_of_B_block[0]; ii < (i+1) * size_of_B_block[0]; ii ++) {
                        for (int jj = j * size_of_B_block[1]; jj < (j+1) * size_of_B_block[1]; jj ++) {
                            temp_block_buffer_B[ii - i * size_of_B_block[0]][jj - j * size_of_B_block[1]] = B[ii][jj];
                        }
                    }
                    MPI_Send(temp_block_buffer_B[0], 
                        size_of_B_block[0] * size_of_B_block[1], 
                        MPI_DOUBLE, 
                        ((i - j + num_of_rows_and_cols_of_procs) % num_of_rows_and_cols_of_procs) * num_of_rows_and_cols_of_procs + j,
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
    if (my_rank == 1) {
        printf("the A block from rank %d is: \n", my_rank);
        log_matrix(A_block, 3,3);
    }
    #endif


    // shifting and calculate

    // gather

    free_matrix(A_block);
    free_matrix(B_block);
    free_matrix(A);
    free_matrix(B);

    MPI_Finalize();
    return 0;
}