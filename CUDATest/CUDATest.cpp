// CUDATest.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include <iostream>
#include <memory>
#include <string>

void checkGpuMem() {
	size_t avail=0;
	size_t total=0;
	cudaError_t rt = cudaMemGetInfo(&avail, &total);
	size_t used = total - avail;
	std::cout << "=================" << std::endl;
	std::cout << "Device memory used: " << used << std::endl;
	std::cout << "Total memory  used: " << total << std::endl;
	std::cout << "=================" << std::endl;
}

int main() {
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
	}
	else {
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}
	checkGpuMem();
	getchar();
}