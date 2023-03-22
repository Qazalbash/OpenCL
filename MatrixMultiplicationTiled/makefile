CC = gcc
CFLAG = -Wall -Wextra
LDFLAG = -lOpenCL
SOURCE = MatrixMultiplicationTiled

$(SOURCE): $(SOURCE).c
	$(CC) $(CFLAG) $(SOURCE).c -o $(SOURCE) $(LDFLAG)

run: $(SOURCE)
	./$(SOURCE)

clean:
	rm -f $(SOURCE_WMB)