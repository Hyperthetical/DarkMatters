LDFLAGS = `gsl-config --cflags --libs` -lm -fopenmp -fPIC -Wshadow 
LDOFLAGS = -fopenmp
CC = g++

default: rebuild

objects: electron.c
	$(CC) -g -c electron.c -o electron.o $(LDFLAGS)

executable: electron.o
	$(CC) -g electron.o -o electron.x $(LDFLAGS)

rebuild: electron.c
	$(CC) -g -c electron.c -o electron.o $(LDFLAGS) 
	$(CC) -g electron.o -o electron.x $(LDFLAGS) 
old: electron_old.c
	$(CC) -g -c electron_old.c -o electron.o $(LDFLAGS) 
	$(CC) -g electron.o -o electron.x $(LDFLAGS) 
exp: electron_exp.c
	$(CC) -g -c electron_exp.c -o electron.o $(LDFLAGS) 
	$(CC) -g electron.o -o electron.x $(LDFLAGS) 

clean:
	$(RM) electron.o
	$(RM) electron.x
