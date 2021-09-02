#pragma once
#include <Utility/STPDeviceErrorHandler.h>

//CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
//System
#include <iostream>
#include <sstream>
#include <typeinfo>

#include <Utility/Exception/STPCUDAError.h>

using std::stringstream;
using std::cerr;
using std::endl;

//Decide what to do next when we catch an error
void programDecision(stringstream& msg, unsigned int error_level, bool no_msg) {
	if (!no_msg) {
		cerr << msg.str();
	}

	switch (error_level) {
	case 0: //STP_CONTINUE_ON_ERROR
		break;
	case 1: //STP_EXCEPTION_ON_ERROR
		throw SuperTerrainPlus::STPException::STPCUDAError(msg.str().c_str());
		break;
	default:
		exit(EXIT_FAILURE);
	}
}

//Helpers to cut down coding efforts
#define ASSERT_FUNCTION(ERR) template<> STP_API void STPcudaAssert<ERR>(ERR cuda_code, unsigned int error_level, const char* __restrict file, const char* __restrict function, int line, bool no_msg)
#define WRITE_ERR_STRING(SS) SS << file << "(" << function << "):" << line
#define CALL_PROGRAM_DECISION(SS) programDecision(SS, error_level, no_msg)

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