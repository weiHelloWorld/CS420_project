SHELL=/bin/bash

CC=mpiicc
CCFLAGS = -std=c99 -O2 -mkl=sequential -openmp

SRC=naive.c MPI_cannon.c MPI_OpenMP_cannon.c
EXE=$(SRC:.c=.exe)

all: $(EXE)

%.exe: %.c support.h
	$(CC) $(CCFLAGS) $< -o $@

clean: 
	rm -f *.o $(EXE) verify.txt

