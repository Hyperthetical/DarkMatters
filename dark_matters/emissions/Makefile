LDFLAGS = -fopenmp -lm -fPIC 
LDOFLAGS = -fopenmp
CC = g++

default: rebuild

objects: electron.c
	$(CC) -c electron.c -o electron.o $(LDFLAGS)

executable: electron.o
	$(CC) electron.o -o electron.x $(LDOFLAGS)

rebuild: electron.c
	$(CC) -c electron.c -o electron.o $(LDFLAGS)
	$(CC) electron.o -o electron.x $(LDOFLAGS)

clean:
	$(RM) electron.o
	$(RM) electron.x
