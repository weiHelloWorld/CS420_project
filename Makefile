SHELL=/bin/bash

CC=mpiicc
CCFLAGS = -std=c99 -O2 -mkl=sequential -openmp

SRC=naive.c MPI_cannon.c MPI_OpenMP_cannon.c
EXE=$(SRC:.c=.exe)
NATIVES=$(SRC:.c=.mic)

all: $(EXE) $(NATIVES)

%.exe: %.c support.h
	$(CC) $(CCFLAGS) $< -o $@

%.mic: %.c
	$(CC) $(CCFLAGS) -mmic $< -o $@

clean: 
	rm -f *.o $(EXE) $(NATIVES) verify.txt

