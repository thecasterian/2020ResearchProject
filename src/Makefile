UNAME = $(shell uname)

CC = mpicc
CFLAGS = -Wall -O3 -Wno-unused-result

ifeq ($(UNAME), Darwin)
	INCLUDE = -I/usr/local/Cellar/glib/2.66.0/include/glib-2.0 -I/usr/local/include
	LDLIBS = -lm -lglib-2.0 -L/usr/local/Cellar/glib/2.66.0/lib -L/usr/local/opt/gettext/lib -lintl
else
	INCLUDE = -I/usr/local/include -I/usr/include/glib-2.0
	LDLIBS = -lm -lglib-2.0 -lnetcdf
endif

HYPRE = /usr/local/lib/libHYPRE.a

TARGET = main

all: $(TARGET)

clean:
	rm -f $(TARGET)
	rm -f *.a
	rm -f *.o

$(TARGET): $(TARGET).c ibm3d.a
	$(CC) $(INCLUDE) $^ $(HYPRE) -o $@ $(LDLIBS) $(CFLAGS)

ibm3d.a: ibm3d_setup.o ibm3d_fracstep.o ibm3d_export.o utils.o geo3d.o
	ar rcs $@ $^

ibm3d_setup.o: ibm3d_setup.c ibm3d_setup.h
	$(CC) $(INCLUDE) -c $< $(LDLIBS) $(CFLAGS)

ibm3d_fracstep.o: ibm3d_fracstep.c ibm3d_fracstep.h
	$(CC) $(INCLUDE) -c $< $(LDLIBS) $(CFLAGS)

ibm3d_export.o: ibm3d_export.c ibm3d_export.h
	$(CC) $(INCLUDE) -c $< $(LDLIBS) $(CFLAGS)

geo3d.o: geo3d.c geo3d.h
	$(CC) $(INCLUDE) -c $< $(LDLIBS) $(CFLAGS)

utils.o: utils.c utils.h
	$(CC) $(INCLUDE) -c $< $(LDLIBS) $(CFLAGS)

test: test.c ibm3d.a
	$(CC) $(INCLUDE) $^ $(HYPRE) -o $@ $(LDLIBS) $(CFLAGS)

channel: channel.c ibm3d.a
	$(CC) $(INCLUDE) $^ $(HYPRE) -o $@ $(LDLIBS) $(CFLAGS)

building: building.c ibm3d.a
	$(CC) $(INCLUDE) $^ $(HYPRE) -o $@ $(LDLIBS) $(CFLAGS)
