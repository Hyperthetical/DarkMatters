g++ -c electron.c -o electron.o -fopenmp -lm -fPIC -L/usr/local/lib
g++ -o electron.x electron.o -fopenmp
