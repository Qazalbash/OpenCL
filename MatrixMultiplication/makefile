CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -l OpenCL
SRC = mult.c
EXEC = mult

$(EXEC): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(EXEC) $(LDFLAGS)

run: $(EXEC)
	./$(EXEC)

clean:
	rm -rf *.o $(EXEC) *.bin *.out