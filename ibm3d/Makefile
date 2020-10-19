UNAME = $(shell uname)

CC = mpicc
CFLAGS = -Wall -O3 -Wno-unused-result -Wno-unused

ifeq ($(UNAME), Darwin)
	INCLUDE = -I/usr/local/Cellar/glib/2.66.0/include/glib-2.0 -I/usr/local/include
	LDLIBS = -lm -lglib-2.0 -L/usr/local/Cellar/glib/2.66.0/lib -L/usr/local/opt/gettext/lib -lintl
else
	INCLUDE = -I/usr/local/include -I/usr/include/glib-2.0
	LDLIBS = -lm -lglib-2.0
endif

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

test: test.c geo3d.o
	$(CC) $(INCLUDE) geo3d.o test.c -o test $(LDLIBS) $(CFLAGS)
