#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
	MPI_Status status;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_of_procs); 


	int num_of_rows_and_cols_of_procs; // num of rows should be equal to num of cols

	double **A, **B; // two input matrices, matrices could be non-squared
	int size_of_matrix_A[2], size_of_matrix_B[2];
	int size_of_A_block[2], size_of_B_block[2];

	size_of_A_block[0] = size_of_matrix_A[0] / num_of_rows_and_cols_of_procs;
	size_of_A_block[1] = size_of_matrix_A[1] / num_of_rows_and_cols_of_procs;
	size_of_B_block[0] = size_of_matrix_B[0] / num_of_rows_and_cols_of_procs;
	size_of_B_block[1] = size_of_matrix_B[1] / num_of_rows_and_cols_of_procs;

	double **A_block, **B_block, **C_block; 

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
							temp_block_buffer[ii - i * size_of_A_block[0]][jj - j * size_of_A_block[1]] = A[ii][jj];
						}
					}
					MPI_Send(temp_block_buffer[0], 
						size_of_A_block[0] * size_of_A_block[1], 
						MPI_DOUBLE, 
						i * num_of_rows_and_cols_of_procs + ((j - i + num_of_rows_and_cols_of_procs) % num_of_rows_and_cols_of_procs),
						0, /* tag = 0 is for A_block */
						MPI_COMM_WORLD
						);
					// secondly, send B block to procesor
					// ((i - j + num_of_rows_and_cols_of_procs) % num_of_rows_and_cols_of_procs, j)
					for (int ii = i * size_of_B_block[0]; ii < (i+1) * size_of_B_block[0]; ii ++) {
						for (int jj = j * size_of_B_block[1]; jj < (j+1) * size_of_B_block[1]; jj ++) {
							temp_block_buffer[ii - i * size_of_B_block[0]][jj - j * size_of_B_block[1]] = A[ii][jj];
						}
					}
					MPI_Send(temp_block_buffer[0], 
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
			MPI_DOUBLE, 0, 1, MPI_STATUS_IGNORE);
		MPI_Recv(B_block[0], 
			size_of_B_block[0] * size_of_B_block[1],
			MPI_DOUBLE, 0, 1, MPI_STATUS_IGNORE);
	}
	// initialization ends here
		


	// shifting and calculate

	// gather

	MPI_Finalize();
	exit(0);
}