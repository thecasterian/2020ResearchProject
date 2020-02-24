all: cavity.c collocated.c
	gcc cavity.c -o cavity -O2
	gcc collocated.c -o collocated -O2