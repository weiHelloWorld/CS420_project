#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
	MPI_Status status;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_of_tasks); 

	
	int num_of_rows_and_cols_of_procs; // num of rows should be equal to num of cols

	double **A, **B; // two input matrices, matrices could be non-squared
	int size_of_matrix_A[2], size_of_matrix_B[2];
	int size_of_A_block[2], size_of_B_block[2];

	size_of_A_block[0] = size_of_matrix_A[0] / num_of_rows_and_cols_of_procs;
	size_of_A_block[1] = size_of_matrix_A[1] / num_of_rows_and_cols_of_procs;
	size_of_B_block[0] = size_of_matrix_B[0] / num_of_rows_and_cols_of_procs;
	size_of_B_block[1] = size_of_matrix_B[1] / num_of_rows_and_cols_of_procs;

	double **A_block, **B_block, **C_block; 

	// for rank 0: get the input of the two matrices



	// initialization, partition into blocks and send to destination process
	for (int i = 0; i < num_of_rows_and_cols_of_procs; i ++) {
		for (int j = 0; j < num_of_rows_and_cols_of_procs; j ++) {
			if (i == 0 && j == 0) {
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

			}
		}
	}


	// shifting and calculate

	// gather

	MPI_Finalize();
	exit(0);
}