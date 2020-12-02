CFLAGS = -O3 -Wall -Wno-unused-result -lm

ibm2d: ibm2d.c
	mpicc -I/usr/local/include ibm2d.c /usr/local/lib/libHYPRE.a -o ibm2d -O3 -Wall -Wno-unused-result -lm

aocc: ibm2d.c
	clang -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -pthread -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -I/usr/local/include ibm2d.c /usr/local/lib/libHYPRE.a -o ibm2d -O3 -Wall -Wno-unused-result -lm

clean:
	rm -rf ibm2d
