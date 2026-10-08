NVCC  = nvcc
ARCH ?= native
FLAGS = -arch=$(ARCH) -O3 -lineinfo -Iinclude

%: %.cu
	$(NVCC) $(FLAGS) $< -o $@