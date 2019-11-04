CC := g++

Serial : 
	$(CC) Serial.cpp -o Serial

clean :
	rm -rf Serial
