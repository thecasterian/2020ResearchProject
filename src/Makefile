UNAME = $(shell uname)

CC = mpicc
CFLAGS = -Wall -Wextra -O3 -Wno-unused-result

# Uncomment if debug mode.
# CFLAGS += -g -rdynamic

ifeq ($(UNAME), Darwin)
	INCLUDE = -I/usr/local/Cellar/glib/2.66.0/include/glib-2.0 -I/usr/local/include
	LDLIBS = -lm -lglib-2.0 -L/usr/local/Cellar/glib/2.66.0/lib -L/usr/local/opt/gettext/lib -lintl
else
	INCLUDE = -I/usr/include/glib-2.0
	LDLIBS = -lm -lglib-2.0 -lnetcdf -lHYPRE
endif

TARGET = main
OBJS = ibm3d_setup.o ibm3d_init.o ibm3d_fracstep.o ibm3d_turb.o ibm3d_export.o utils.o geo3d.o

all: $(TARGET)

clean:
	rm -f $(TARGET)
	rm -f *.o

$(TARGET): $(TARGET).c $(OBJS)
	$(CC) $(INCLUDE) $^ -o $@ $(LDLIBS) $(CFLAGS)

%.o: %.c %.h
	$(CC) $(INCLUDE) -c $< $(LDLIBS) $(CFLAGS)

test: test.c $(OBJS)
	$(CC) $(INCLUDE) $^ -o $@ $(LDLIBS) $(CFLAGS)

channel: channel.c $(OBJS)
	$(CC) $(INCLUDE) $^ -o $@ $(LDLIBS) $(CFLAGS)

building: building.c $(OBJS)
	$(CC) $(INCLUDE) $^ -o $@ $(LDLIBS) $(CFLAGS)

jet: jet.c $(OBJS)
	$(CC) $(INCLUDE) $^ -o $@ $(LDLIBS) $(CFLAGS)

cavity: cavity.c $(OBJS)
	$(CC) $(INCLUDE) $^ -o $@ $(LDLIBS) $(CFLAGS)
