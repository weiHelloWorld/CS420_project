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

    int my_rank, num_of_procs;

    int required = MPI_THREAD_FUNNELED; // MPI funcs are called by master thread
    int provided;

    MPI_Init_thread(&argc, &argv, required, &provided);
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

    double **A_row, **B_row, **C_row; 

    A_row = create_matrix(size_of_A[0], size_of_A_block[1]);
    B_row = create_matrix(size_of_B[0], size_of_B_block[1]);
    C_row = create_matrix(size_of_A[0], size_of_B_block[1]);

    // send blocks to specific row in specific processor

    // multiply

    // shifting

    // gather


    free_matrix(A_row);
    free_matrix(B_row);
    free_matrix(C_row);

    MPI_Finalize();
    return 0;
}

