#executable name
NAME_MAT = creatrix
NAME_SERIAL = hungarian_serial
#flags
CPPFLAGS += -g -O3 -std=c++11
all: $(NAME_MAT) $(NAME_SERIAL)

#making Serial program : Serial.cpp
$(NAME_SERIAL): $(NAME_SERIAL).cpp
	$(CXX) $(CPP_FLAGS) $^ -o $(NAME_SERIAL)

#making matrix creation program : creatrix.cpp
$(NAME_MAT): $(NAME_MAT).cpp
	$(CXX) $(CPP_FLAGS) $^ -o $(NAME_MAT)

clean:
	rm $(NAME_MAT) $(NAME_SERIAL)