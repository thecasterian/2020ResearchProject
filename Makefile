all: staggered uniform nonuniform

staggered: cavity_staggered.c
	gcc cavity_staggered.c -o cavity_staggered -O3 -lm

uniform: cavity_collocated.c
	gcc cavity_collocated.c -o cavity_collocated -O3 -lm

nonuniform: cavity_collocated_nonuniform.c
	gcc cavity_collocated_nonuniform.c -o cavity_collocated_nonuniform -O3 -lm
