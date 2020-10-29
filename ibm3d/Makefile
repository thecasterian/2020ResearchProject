UNAME = $(shell uname)

CC = mpicc
CFLAGS = -Wall -O3 -Wno-unused-result -Wno-incompatible-pointer-types

ifeq ($(UNAME), Darwin)
	INCLUDE = -I/usr/local/Cellar/glib/2.66.0/include/glib-2.0 -I/usr/local/include
	LDLIBS = -lm -lglib-2.0 -L/usr/local/Cellar/glib/2.66.0/lib -L/usr/local/opt/gettext/lib -lintl
else
	INCLUDE = -I/usr/local/include -I/usr/include/glib-2.0
	LDLIBS = -lm -lglib-2.0
endif

HYPRE = /usr/local/lib/libHYPRE.a

TARGET = main

all: $(TARGET)

clean:
	rm -f $(TARGET)
	rm -f ibm3d.o
	rm -f geo3d.o

$(TARGET): main.c ibm3d.a geo3d.o
	$(CC) $(INCLUDE) $^ $(HYPRE) -o $@ $(LDLIBS) $(CFLAGS)

ibm3d.a: ibm3d_setup.o ibm3d_fracstep.o
	ar rcs $@ $^

ibm3d_setup.o: ibm3d_setup.c
	$(CC) $(INCLUDE) -c $< $(LDLIBS) $(CFLAGS)

ibm3d_fracstep.o: ibm3d_fracstep.c
	$(CC) $(INCLUDE) -c $< $(LDLIBS) $(CFLAGS)

geo3d.o: geo3d.h geo3d.c
	$(CC) $(INCLUDE) -c geo3d.c $(LDLIBS) $(CFLAGS)

test: test.c geo3d.o
	$(CC) $(INCLUDE) geo3d.o test.c -o test $(LDLIBS) $(CFLAGS)
