run:
		mpiexec -n 6 main lena200_noisy.txt output.txt 0.6 0.1
		python text_to_image.py output.txt lena200_out.png
all: main.cpp
		mpic++ main.cpp -o main
