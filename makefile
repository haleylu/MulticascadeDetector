CC = g++
CFLAGS = -g -Wall
SRCS = main.cpp MulticascadeDetector.cpp
OBJS = $(SRCS : .cpp = .o)
PROG = cascade_test

LIBS = $(OPENCV)
OPENCV =`pkg-config opencv --cflags --libs`


$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
	
.PHONY: clean
clean: 
	rm -f *.o cascade_test
	
