DIRS = src

.PHONY: all clean

all:
	@for d in $(DIRS); \
	do \
		cd $$d; $(MAKE) all; \
	done

clean:
	rm -rf build

main: main.c all
	mpicc main.c -o main -lm -lglib-2.0 -L./build -libm3d -lnetcdf -lHYPRE -g -rdynamic
