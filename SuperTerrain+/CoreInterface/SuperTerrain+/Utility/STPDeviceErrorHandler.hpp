#include <SuperTerrain+/Utility/STPGenericErrorHandlerBlueprint.hpp>

/* STPDeviceErrorHandler contains some error handlers for CUDA API calls.
   To eliminate header pollution, we force users to include the corresponding CUDA headers to enable their error handling function. */

#ifdef __DRIVER_TYPES_H__
#define STP_HAS_CUDA_RUNTIME
#endif
#ifdef __cuda_cuda_h__
#define STP_HAS_CUDA_DRIVER
#endif
#ifdef __NVRTC_H__
#define STP_HAS_CUDA_NVRTC
#endif

//#include <cuda_runtime.h>
#ifdef STP_HAS_CUDA_RUNTIME
#ifndef _STP_CUDA_RUNTIME_ERROR_HANDLER_HPP_
#define _STP_CUDA_RUNTIME_ERROR_HANDLER_HPP_
#include <SuperTerrain+/Exception/STPCUDAError.h>

STP_ERROR_DESCRIPTOR(assertCUDA, cudaError_t, cudaSuccess, STPException::STPCUDAError) {
	msg_str << "CUDA Runtime API: " << cudaGetErrorString(error_code);
}
#endif//_STP_CUDA_RUNTIME_ERROR_HANDLER_HPP_
#endif//STP_HAS_CUDA_RUNTIME

//#include <cuda.h>
#ifdef STP_HAS_CUDA_DRIVER
#ifndef _STP_CUDA_DRIVER_ERROR_HANDLER_HPP_
#define _STP_CUDA_DRIVER_ERROR_HANDLER_HPP_
#include <SuperTerrain+/Exception/STPCUDAError.h>

STP_ERROR_DESCRIPTOR(assertCUDA, CUresult, CUDA_SUCCESS, STPException::STPCUDAError) {
	const char* str;
	cuGetErrorString(error_code, &str);
	msg_str << "CUDA Driver API: " << str;
}
#endif//_STP_CUDA_DRIVER_ERROR_HANDLER_HPP_
#endif//STP_HAS_CUDA_DRIVER

//#include <nvrtc.h>
#ifdef STP_HAS_CUDA_NVRTC
#ifndef _STP_CUDA_NVRTC_ERROR_HANDLER_HPP_
#define _STP_CUDA_NVRTC_ERROR_HANDLER_HPP_
#include <SuperTerrain+/Exception/STPCUDAError.h>

STP_ERROR_DESCRIPTOR(assertCUDA, nvrtcResult, NVRTC_SUCCESS, STPException::STPCUDAError) {
	msg_str << "CUDA Runtime Compiler: " << nvrtcGetErrorString(error_code);
}
#endif//_STP_CUDA_NVRTC_ERROR_HANDLER_HPP_
#endif//STP_HAS_CUDA_NVRTC

#if defined(STP_HAS_CUDA_RUNTIME) || defined(STP_HAS_CUDA_DRIVER) || defined(STP_HAS_CUDA_NVRTC)
#ifndef _STP_DEVICE_ERROR_HANDLER_HPP_
#define _STP_DEVICE_ERROR_HANDLER_HPP_

#define STP_CHECK_CUDA(ERR) STP_INVOKE_ERROR_DESCRIPTOR(assertCUDA, ERR)

#endif//_STP_DEVICE_ERROR_HANDLER_HPP_
#endif

#undef STP_HAS_CUDA_RUNTIME
#undef STP_HAS_CUDA_DRIVER
#undef STP_HAS_CUDA_NVRTC