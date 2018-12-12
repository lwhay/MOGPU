a.out : main.o kernel.o
	nvcc -arch=sm_35 main.o kernel.o -lcudadevrt -o a.out

main.o : main.cu kernel.h
	nvcc -Xptxas -v -arch=sm_35 -dc -g -G main.cu

kernel.o : kernel.cu kernel.h
	nvcc -Xptxas -v -arch=sm_35 -dc -g -G kernel.cu

clean : 
	rm -rf *.o 


