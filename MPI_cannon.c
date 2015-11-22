#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>   
#include "support.h"

#define DEBUG

void my_print_matrix (char* filename, double **mat, int m, int n) {
    FILE* fp;
    int i, j;

    fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Unable to write the file %s\n", filename);
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

int main(int argc, char* argv[]) {
    MPI_Status status;
    MPI_Request request[4];
    int my_rank, num_of_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_procs); 

    int row_num_of_procs; // num of rows should be equal to num of cols

    double **A, **B; // two input matrices, matrices could be non-squared
    double **C; // the result
    int size_of_A[2], size_of_B[2];
    int size_of_A_block[2], size_of_B_block[2];

    #ifdef DEBUG
    A = create_matrix(9, 9);
    B = create_matrix(9, 9);
    init_spec(A, 9, 9);
    init_spec(B, 9, 9);
    size_of_A[0] = 9; size_of_A[1] = 9;
    size_of_B[0] = 9; size_of_B[1] = 9;
    row_num_of_procs = 3;
    if (my_rank == 0) {
        my_print_matrix("matrix_A.txt", A, 9, 9);
        my_print_matrix("matrix_B.txt", B, 9, 9);
    }

    #endif

    C = create_matrix(size_of_A[0], size_of_B[1]);

    size_of_A_block[0] = size_of_A[0] / row_num_of_procs;
    size_of_A_block[1] = size_of_A[1] / row_num_of_procs;
    size_of_B_block[0] = size_of_B[0] / row_num_of_procs;
    size_of_B_block[1] = size_of_B[1] / row_num_of_procs;

    assert(size_of_A_block[1] == size_of_B_block[0]);

    double **A_block, **B_block, **C_block, **temp_block_buffer_A, **temp_block_buffer_B; 
    A_block = create_matrix(size_of_A_block[0], size_of_A_block[1]);
    B_block = create_matrix(size_of_B_block[0], size_of_B_block[1]);
    C_block = create_matrix(size_of_A_block[0], size_of_B_block[1]);

    init_zero(C_block, size_of_A_block[0], size_of_B_block[1]); 


    temp_block_buffer_A = create_matrix(size_of_A_block[0], size_of_A_block[1]);
    temp_block_buffer_B = create_matrix(size_of_B_block[0], size_of_B_block[1]);


    // for rank 0: get the input of the two matrices (maybe from files)


    // initialization, partition into blocks and send to destination process
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
                    #ifdef DEBUG
                    // printf("temp_block_buffer_A = \n");
                    // log_matrix(temp_block_buffer_A, 3, 3);
                    assert(num_of_procs == 9);
                    #endif

                    assert(size_of_A_block[0] * size_of_A_block[1] == num_of_procs);
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
        for (int i = 0; i < size_of_A_block[0]; i ++) {
            for (int j = 0; j < size_of_A_block[1]; j ++) {
                for (int k = 0; k < size_of_B_block[1]; k ++) {
                    C_block[i][k] += A_block[i][j] * B_block[j][k];
                }
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

    #ifdef DEBUG
    if (my_rank == 0) {
            printf("the C from rank %d is: \n", my_rank);
            log_matrix(C, 9,9);
        }
    #endif

    free_matrix(A_block);
    free_matrix(B_block);
    free_matrix(A);
    free_matrix(B);
    free_matrix(temp_for_gather);
    free_matrix(temp_block_buffer_A);
    free_matrix(temp_block_buffer_B);


    MPI_Finalize();
    return 0;
}

