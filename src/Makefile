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
OBJS = ibm3d_setup.o ibm3d_init.o ibm3d_fracstep.o ibm3d_export.o utils.o geo3d.o

all: $(TARGET)

clean:
	rm -f $(TARGET)
	rm -f *.a
	rm -f *.o

$(TARGET): $(TARGET).c $(OBJS)
	$(CC) $(INCLUDE) $^ $(HYPRE) -o $@ $(LDLIBS) $(CFLAGS)

%.o: %.c %.h
	$(CC) $(INCLUDE) -c $< $(LDLIBS) $(CFLAGS)

test: test.c $(OBJS)
	$(CC) $(INCLUDE) $^ $(HYPRE) -o $@ $(LDLIBS) $(CFLAGS)

channel: channel.c $(OBJS)
	$(CC) $(INCLUDE) $^ $(HYPRE) -o $@ $(LDLIBS) $(CFLAGS)

building: building.c $(OBJS)
	$(CC) $(INCLUDE) $^ $(HYPRE) -o $@ $(LDLIBS) $(CFLAGS)
