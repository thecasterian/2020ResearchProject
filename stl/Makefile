CC = gcc
CFLAGS = -Wall -O2
INCLUDE = -I/usr/include/glib-2.0
LDLIBS = -lm -lglib-2.0

TARGET = stl_reader

all: $(TARGET)

clean:
	rm -f $(TARGET)

$(TARGET): stl_reader.c
	$(CC) $(INCLUDE) stl_reader.c -o $@ $(LDLIBS) $(CFLAGS)
