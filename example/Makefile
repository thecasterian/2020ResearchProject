CC = mpicc
CFLAGS = -Wall -O3

INCLUDE = -I/usr/local/include
LDLIBS = -lm
HYPRE = /usr/local/lib/libHYPRE.a

IJsolver: IJsolver.c
	$(CC) $(INCLUDE) $^ $(HYPRE) -o $@ $(LDLIBS) $(CFLAGS)

ex5: ex5.c
	$(CC) $(INCLUDE) $^ $(HYPRE) -o $@ $(LDLIBS) $(CFLAGS)

ex3: ex3.c
	$(CC) $(INCLUDE) $^ $(HYPRE) -o $@ $(LDLIBS) $(CFLAGS)

netcdf: netcdf.c
	gcc netcdf.c -o netcdf -lnetcdf $(CFLAGS)

clean:
	rm -rf IJsolver
	rm -rf ex5
	rm -rf ex3
