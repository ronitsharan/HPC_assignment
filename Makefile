CC = gcc
CFLAGS = -O3 -fopenmp -mavx -fPIC -I.
LDLIBS = -lpthread -lm -fopenmp -mkl=parallel -Werror -Wall -pedantic

test:
	$(CC) $(CFLAGS) test.cpp $(LDLIBS)

clean:
	@rm -f a.out
