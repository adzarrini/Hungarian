#executable name
NAME_MAT = creatrix
NAME_SERIAL = hungarian_serial
NAME_PARALLEL = hungarian_parallel
#compiler
NVCC = nvcc
#flags
CPPFLAGS += -g -O3 -std=c++11
CUDAFLAGS = -O3 -arch=sm_35 -D_FORCE_INLINES 

all: $(NAME_MAT) $(NAME_SERIAL) $(NAME_PARALLEL)

#making Parallel program : hungarian_parallel.cpp
$(NAME_PARALLEL): $(NAME_PARALLEL).cu
	$(NVCC) -o $(NAME_PARALLEL) $^ $(CUDAFLAGS)  

#making Serial program : hungarian_serial.cpp
$(NAME_SERIAL): $(NAME_SERIAL).cpp
	$(CXX) $(CPP_FLAGS) $^ -o $(NAME_SERIAL)

#making matrix creation program : creatrix.cpp
$(NAME_MAT): $(NAME_MAT).cpp
	$(CXX) $(CPP_FLAGS) $^ -o $(NAME_MAT)

clean:
	rm $(NAME_MAT) $(NAME_SERIAL)
