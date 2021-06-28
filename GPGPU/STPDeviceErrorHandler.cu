#include "STPDeviceErrorHandler.cuh"

//System
#include <iostream>

__host__ inline void SuperTerrainPlus::STPCompute::cudaAssert(cudaError_t cuda_code, const char* file, int line) {
	if (cuda_code != cudaSuccess) {
		std::cerr << "CUDA assert: " << cudaGetErrorString(cuda_code) << " in " << file << " at line " << line << std::endl;
	}
}