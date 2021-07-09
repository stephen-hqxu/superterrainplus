#pragma once
#ifndef _STP_DEVICE_ERROR_HANDLER_CUH_
#define _STP_DEVICE_ERROR_HANDLER_CUH_

//CUDA
#include <cuda_runtime.h>
//System
#include <iostream>
#include <sstream>

__host__ inline void STPcudaAssert(cudaError_t cuda_code, const char* __restrict file, int line) {
	using std::cerr;
	using std::endl;

	if (cuda_code != cudaSuccess) {
		std::stringstream err_str;
		err_str << file << ":" << line << "\nError: " << cudaGetErrorString(cuda_code) << endl;
		cerr << err_str.str();

#ifndef STP_CONTINUE_ON_ERROR
#ifdef STP_EXCEPTION_ON_ERROR
#include <stdexcept>
#include <exception>
		throw std::runtime_error(err_str.str());
#else
		exit(EXIT_FAILURE);
#endif //STP_EXCEPTION_ON_ERROR
#endif //STP_CONTINUE_ON_ERROR
	}
}

#define STPcudaCheckErr(ans) STPcudaAssert(ans, __FILE__, __LINE__);
#endif//_STP_DEVICE_ERROR_HANDLER_CUH_