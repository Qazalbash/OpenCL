CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -l OpenCL
SRC = add.c
EXEC = add

$(EXEC): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(EXEC) $(LDFLAGS)

run: $(EXEC)
	./$(EXEC)

clean:
	rm -rf *.o $(EXEC) *.bin *.out