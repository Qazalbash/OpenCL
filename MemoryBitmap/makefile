CC = gcc
CFLAG = -Wall -Wextra
LDFLAG = -lOpenCL
SOURCE = SharedMemoryBitmap

$(SOURCE): $(SOURCE).c
	$(CC) $(CFLAG) $(SOURCE).c -o $(SOURCE) $(LDFLAG)

run: $(SOURCE)
	./$(SOURCE)

clean:
	rm -f $(SOURCE)