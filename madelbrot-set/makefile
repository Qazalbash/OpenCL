all: set.c kernel.cl
	gcc set.c -Wall -Wextra -o set -l OpenCL

run: all
	./set
	code mandelbrot.ppm

clean: set
	rm ./set