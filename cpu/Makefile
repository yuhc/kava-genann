CFLAGS = -Wall -Wshadow -O3 -g -march=native -fPIC -I.
LDLIBS = -L. -lm -lgenann -Wl,-rpath=.

all: lib check example1 example2 example3 example4 runmnist
lib: libgenann

sigmoid: CFLAGS += -Dgenann_act=genann_act_sigmoid_cached
sigmoid: all

threshold: CFLAGS += -Dgenann_act=genann_act_threshold
threshold: all

linear: CFLAGS += -Dgenann_act=genann_act_linear
linear: all

libgenann: genann.o
	cc   -shared -o libgenann.so genann.o -lm

test: test.o

check: test
	./$^

example1: example1.o

example2: example2.o

example3: example3.o

example4: example4.o

runmnist: mnist/runmnist.o mnist/mnist.o
	cc   mnist/runmnist.o mnist/mnist.o ${LDLIBS} -o runmnist


clean:
	$(RM) *.o mnist/*.o
	$(RM) *.so
	$(RM) test example1 example2 example3 example4 *.exe runmnist
	$(RM) persist.txt

.PHONY: sigmoid threshold linear clean
