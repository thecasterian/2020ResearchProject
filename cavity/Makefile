CFLAGS = -O3 -Wall -Wno-unused-result -lm

all: staggered uniform nonuniform

staggered: cavity_staggered.c
	gcc cavity_staggered.c -o cavity_staggered $(CFLAGS)

uniform: cavity_collocated.c
	gcc cavity_collocated.c -o cavity_collocated $(CFLAGS)

nonuniform: cavity_collocated_nonuniform.c
	gcc cavity_collocated_nonuniform.c -o cavity_collocated_nonuniform $(CFLAGS)

uniform_hypre: cavity_collocated_hypre.c
	mpicc -I/usr/local/include cavity_collocated_hypre.c /usr/local/lib/libHYPRE.a -o cavity_collocated_hypre $(CFLAGS)

nonuniform_hypre: cavity_collocated_nonuniform_hypre.c
	mpicc -I/usr/local/include cavity_collocated_nonuniform_hypre.c /usr/local/lib/libHYPRE.a -o cavity_collocated_nonuniform_hypre $(CFLAGS)

mpi: cavity_mpi.c
	mpicc -I/usr/local/include cavity_mpi.c /usr/local/lib/libHYPRE.a -o cavity_mpi $(CFLAGS)

clean:
	rm -rf cavity_staggered
	rm -rf cavity_collocated
	rm -rf cavity_collocated_nonuniform
	rm -rf cavity_collocated_hypre
	rm -rf cavity_collocated_nonuniform_hypre
	rm -rf cavity_mpi
