#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
	MPI_Status status;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_of_tasks); 

	
	int num_of_rows_and_cols_of_procs[2];  // currently we need to assume that both sizes of A and B are divisible by this

	double **A, **B; // two input matrices
	int size_of_matrix_A[2], size_of_matrix_B[2];

	double **A_block, **B_block, **C_block; 

	// for rank 0: get the input of the two matrices
	




	// initialization, partition into blocks and send to destination process

	// shifting and calculate

	// gather
}