all: staggered uniform nonuniform

staggered: cavity_staggered.c
	gcc cavity_staggered.c -o cavity_staggered -O3 -lm

uniform: cavity_collocated.c
	gcc cavity_collocated.c -o cavity_collocated -O3 -lm

nonuniform: cavity_collocated_nonuniform.c
	gcc cavity_collocated_nonuniform.c -o cavity_collocated_nonuniform -O3 -Wall -lm

uniform_hypre: cavity_collocated_hypre.c
	mpicc -I/usr/apps/include cavity_collocated_hypre.c /usr/apps/lib/libHYPRE.a -o cavity_collocated_hypre -lm

nonuniform_hypre: cavity_collocated_nonuniform_hypre.c
	mpicc -I/usr/apps/include cavity_collocated_nonuniform_hypre.c /usr/apps/lib/libHYPRE.a -o cavity_collocated_nonuniform_hypre -lm