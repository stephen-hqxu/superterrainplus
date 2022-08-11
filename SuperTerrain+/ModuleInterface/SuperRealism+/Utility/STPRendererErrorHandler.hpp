#include <SuperTerrain+/Utility/STPGenericErrorHandlerBlueprint.hpp>

#ifdef __optix_optix_types_h__
#define STP_HAS_OPTIX
#endif

//#include <optix.h>
#ifdef STP_HAS_OPTIX
#ifndef _STP_RENDERER_ERROR_HANDLER_HPP_
#define _STP_RENDERER_ERROR_HANDLER_HPP_
#include <optix_stubs.h>
//well OptiX is based on CUDA, so we just boil it down to CUDA error.
#include <SuperTerrain+/Exception/STPCUDAError.h>

STP_ERROR_DESCRIPTOR(assertOptiX, OptixResult, OPTIX_SUCCESS, STPException::STPCUDAError) {
	msg_str << "OptiX: " << optixGetErrorString(error_code);
}
#define STP_CHECK_OPTIX(ERR) STP_INVOKE_ERROR_DESCRIPTOR(assertOptiX, ERR)

#endif//_STP_RENDERER_ERROR_HANDLER_HPP_
#endif//STP_HAS_OPTIX

#undef STP_HAS_OPTIX