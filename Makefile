MEG: ME_GPU.cu
	nvcc -arch=sm_52 -ccbin clang-3.8 ME_GPU.cu -o MEG
