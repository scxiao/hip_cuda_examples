HIP_PATH=/opt/rocm/hip
HIPCC=$(HIP_PATH)/bin/hipcc


CPPFLAGS=-c -O2 -std=c++11
NVCCFLAGS=-c -O2 -std=c++11 
#NVCCFLAGS=-c -O2 -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__
LDDFLAGS= -lpthread -lcudart -L/usr/local/cuda/lib64
CXX=g++
NVCC=nvcc

SRC=.
OBJ=obj
EXE=.
OBJFILES=$(OBJ)/test_mm_mul.o \
         $(OBJ)/cu_matrix_mul.o \
         $(OBJ)/timer.o

all: create \
     test_mm_mul

test_mm_mul : $(OBJFILES)
	$(CXX) -o $@ $^ $(LDDFLAGS)

$(OBJ)/%.o : $(SRC)/%.cpp
	$(CXX) $(CPPFLAGS) -o $@ $<

$(OBJ)/%.o : $(SRC)/%.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

create:
	if [ ! -d "./obj" ]; then \
		mkdir obj; \
	fi

clean:
	rm -rf 	$(OBJ) $(EXE)/test_mm_mul
