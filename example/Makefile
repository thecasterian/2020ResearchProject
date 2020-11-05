CC = mpicc
CFLAGS = -Wall -O3

INCLUDE = -I/usr/local/include
LDLIBS = -lm
HYPRE = /usr/local/lib/libHYPRE.a

IJsolver: IJsolver.c
	$(CC) $(INCLUDE) $^ $(HYPRE) -o $@ $(LDLIBS) $(CFLAGS)
