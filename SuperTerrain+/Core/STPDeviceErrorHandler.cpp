#pragma once
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

//CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
//System
#include <iostream>
#include <sstream>
#include <typeinfo>

#include <SuperTerrain+/Utility/Exception/STPCUDAError.h>

using std::stringstream;
using std::cerr;
using std::endl;

//always throw an exception
inline void printError(stringstream& msg, bool no_msg) noexcept(false) {
	if (!no_msg) {
		cerr << msg.str();
	}
	//throw exception
	throw SuperTerrainPlus::STPException::STPCUDAError(msg.str().c_str());
}

//Helpers to cut down coding efforts
#define ASSERT_FUNCTION(ERR) template<> STP_API void STPcudaAssert<ERR>(ERR cuda_code, const char* __restrict file, const char* __restrict function, int line, bool no_msg) noexcept(false)
#define WRITE_ERR_STRING(SS) SS << file << "(" << function << "):" << line
#define CALL_PROGRAM_DECISION(SS) printError(SS, no_msg)

//explicit instantiation
ASSERT_FUNCTION(cudaError_t) {
	//CUDA Runtime API
	if (cuda_code != cudaSuccess) {
		stringstream err_str;
		WRITE_ERR_STRING(err_str) << "\nCUDA Runtime API: " << cudaGetErrorString(cuda_code) << endl;
		CALL_PROGRAM_DECISION(err_str);
	}
}

ASSERT_FUNCTION(nvrtcResult) {
	//CUDA Runtime Compilication
	if (cuda_code != NVRTC_SUCCESS) {
		stringstream err_str;
		WRITE_ERR_STRING(err_str) << "\nCUDA Runtime Compiler: " << nvrtcGetErrorString(cuda_code) << endl;
		CALL_PROGRAM_DECISION(err_str);
	}
}

ASSERT_FUNCTION(CUresult) {
	//CUDA Driver API
	if (cuda_code != CUDA_SUCCESS) {
		stringstream err_str;
		const char* str;
		cuGetErrorString(cuda_code, &str);
		WRITE_ERR_STRING(err_str) << "\nCUDA Driver API: " << str << endl;
		CALL_PROGRAM_DECISION(err_str);
	}
}