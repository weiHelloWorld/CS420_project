SHELL=/bin/bash

CC=mpiicc
CCFLAGS = -std=c99 -O2

SRC=naive.c MPI_cannon.c
EXE=$(SRC:.c=.exe)

all: $(EXE)

%.exe: %.c
	$(CC) $(CCFLAGS) $< -o $@

clean: 
	rm -f *.o $(EXE) verify.txt
