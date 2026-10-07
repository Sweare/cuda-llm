NVCC  = nvcc
ARCH ?= native
FLAGS = -arch=$(ARCH) -O3 -lineinfo

%: %.cu
	$(NVCC) $(FLAGS) $< -o $@