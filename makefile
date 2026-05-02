# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Wextra -O2

# Target executable
TARGET = matrix_mult_tile 

# Source files
SRC = matrix_mult_tile.c 

# Default target
all: $(TARGET)

# Rule for the GPU version
matrix_mult_tile: matrix_mult_tile.c
	$(CC) $(CFLAGS) -o matrix_mult_tile matrix_mult_tile.c $(LDFLAGS)

# Clean up build artifacts
clean:
	rm -f $(TARGET)

