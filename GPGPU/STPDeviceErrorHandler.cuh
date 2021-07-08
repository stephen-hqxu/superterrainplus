#pragma once
#ifndef _STP_DEVICE_ERROR_HANDLER_CUH_
#define _STP_DEVICE_ERROR_HANDLER_CUH_

//CUDA
#include <cuda_runtime.h>
//System
#include <iostream>
#include <sstream>

__host__ inline void STPcudaAssert(cudaError_t cuda_code, const char* restrict file, int line) {
	using std::cerr;
	using std::endl;

	if (cuda_code != cudaSuccess) {
		std::stringstream err_str;
		err_str << file << "(" << line << ")::" << cudaGetErrorString(cuda_code) << endl;
		cerr << err_str.str();

#ifdef STP_EXIT_ON_ERROR
		exit(EXIT_FAILURE);
#elif defined STP_EXCEPTION_ON_ERROR//STP_EXIT_ON_ERROR
		throw std::runtime_error(err_str.str());
#endif//STP_EXCEPTION_ON_ERROR

	}
}

#define STPcudaCheckErr(ans) {STPcudaAssert(ans, __FILE__, __LINE__);}
#endif//_STP_DEVICE_ERROR_HANDLER_CUH_