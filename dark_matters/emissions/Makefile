LDFLAGS = `gsl-config --cflags --libs` -lm -fopenmp -fPIC -Wshadow 
LDOFLAGS = -fopenmp
CC = g++

default: rebuild

objects: electron.c
	$(CC) -c electron.c -o electron.o $(LDFLAGS)

executable: electron.o
	$(CC) electron.o -o electron.x $(LDFLAGS)

rebuild: electron.c
	$(CC) -c electron.c -o electron.o $(LDFLAGS) 
	$(CC) electron.o -o electron.x $(LDFLAGS) 
clean:
	$(RM) electron.o
	$(RM) electron.x
