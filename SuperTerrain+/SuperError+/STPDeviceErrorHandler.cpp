#pragma once
#include <STPDeviceErrorHandler.h>

//CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
//System
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>

using std::stringstream;
using std::cerr;
using std::endl;

//Decide what to do next when we catch an error
void programDecision(stringstream& msg, unsigned int error_level) {
	cerr << msg.str();

	switch (error_level) {
	case 0: //STP_CONTINUE_ON_ERROR
		break;
	case 1: //STP_EXCEPTION_ON_ERROR
		throw std::runtime_error(msg.str());
		break;
	default:
		exit(EXIT_FAILURE);
	}
}

//explicit instantiation
template<> STPERRORPLUS_API void STPcudaAssert<cudaError_t>(cudaError_t cuda_code, unsigned int error_level, const char* __restrict file, const char* __restrict function, int line) {
	//CUDA Runtime API
	if (cuda_code != cudaSuccess) {
		stringstream err_str;
		err_str << file << "(" << function << "):" << line << "\nCUDA Runtime API: " << cudaGetErrorString(cuda_code) << endl;
		programDecision(err_str, error_level);
	}
}

template<> STPERRORPLUS_API void STPcudaAssert<nvrtcResult>(nvrtcResult cuda_code, unsigned int error_level, const char* __restrict file, const char* __restrict function, int line) {
	//CUDA Runtime Compilication
	if (cuda_code != NVRTC_SUCCESS) {
		stringstream err_str;
		err_str << file << "(" << function << "):" << line << "\nCUDA Runtime Compiler: " << nvrtcGetErrorString(cuda_code) << endl;
		programDecision(err_str, error_level);
	}
}

template<> STPERRORPLUS_API void STPcudaAssert<CUresult>(CUresult cuda_code, unsigned int error_level, const char* __restrict file, const char* __restrict function, int line) {
	//CUDA Driver API
	if (cuda_code != CUDA_SUCCESS) {
		stringstream err_str;
		const char* str;
		cuGetErrorString(cuda_code, &str);
		err_str << file << "(" << function << "):" << line << "\nCUDA Driver API: " << str << endl;
		programDecision(err_str, error_level);
	}
	
}