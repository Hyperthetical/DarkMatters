g++ -c electron.c -o electron.o -fopenmp -lm -fPIC
g++ electron.o -o electron.x -fopenmp
