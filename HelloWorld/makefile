CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -l OpenCL
SRC = hello.c
EXEC = hello

$(EXEC): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(EXEC) $(LDFLAGS)

run: $(EXEC)
	./$(EXEC)

clean:
	rm -rf *.o $(EXEC) *.bin *.out