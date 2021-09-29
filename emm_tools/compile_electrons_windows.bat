g++ -c electron.c -o electron_win.o -fopenmp -lm -fPIC -L"C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64"
g++ -o electron.exe electron_win.o -fopenmp
