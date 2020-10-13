CFLAGS = -O3 -Wall -Wno-unused-result -lm

ibm2d: ibm2d.c
	mpicc -I/usr/local/include ibm2d.c /usr/local/lib/libHYPRE.a -o ibm2d $(CFLAGS)

clean:
	rm -rf ibm2d
