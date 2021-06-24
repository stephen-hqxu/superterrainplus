#pragma once
#ifndef _STP_DEVICE_ERROR_HANDLER_CUH_
#define _STP_DEVICE_ERROR_HANDLER_CUH_

//System
#include <iostream>
//CUDA
#include <cuda_runtime.h>

//unfinished, need to refine the error handling later
#define cudaCheckError(ans) {cudaAssert(ans, __FILE__, __LINE__)}
__host__ inline void cudaAssert(cudaError_t cuda_code, const char* file, int line) {
	if (cuda_code != cudaSuccess) {
		std::cerr << "CUDA assert: " << cudaGetErrorString(cuda_code) << " in " << file << " at " << line << std::endl;
	}
}

#endif//_STP_DEVICE_ERROR_HANDLER_CUH_