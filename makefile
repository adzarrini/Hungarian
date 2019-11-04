#executable name
NAME_MAT = creatrix
NAME_SERIAL = Serial
#flags 
CPPFLAGS += -g -O3 -std=c++11 
all: $(NAME_MAT) $(NAME_SERIAL)

#making hw1 program
$(NAME_SERIAL): $(NAME_SERIAL).cpp 
	$(CXX) $(CPP_FLAGS) $^ -o $(NAME_SERIAL) 

#making matrix creation file
$(NAME_MAT): $(NAME_MAT).cpp
	$(CXX) $(CPP_FLAGS) $^ -o $(NAME_MAT)

clean:
	rm $(NAME_MAT) $(NAME_SERIAL) matrix/*.txt
