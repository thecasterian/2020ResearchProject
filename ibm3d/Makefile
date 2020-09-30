CC = mpicc
CFLAGS = -Wall -Wno-unused-result -O2
INCLUDE = -I/usr/include/glib-2.0 -I/usr/local/include
LDLIBS = -lm -lglib-2.0

HYPRE = /usr/local/lib/libHYPRE.a

TARGET = ibm3d

all: $(TARGET)

clean:
	rm -f $(TARGET)
	rm -f geo3d.o

$(TARGET): ibm3d.c geo3d.o
	$(CC) $(INCLUDE) ibm3d.c geo3d.o $(HYPRE) -o $@ $(LDLIBS) $(CFLAGS)

geo3d.o: geo3d.h geo3d.c
	$(CC) $(INCLUDE) -c geo3d.c $(LDLIBS) $(CFLAGS)
