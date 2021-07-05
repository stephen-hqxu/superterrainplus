#pragma once
#ifndef _STP_DEVICE_ERROR_HANDLER_CUH_
#define _STP_DEVICE_ERROR_HANDLER_CUH_

//CUDA
#include <cuda_runtime.h>

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {
	/**
	 * @brief GPGPU compute suites for Super Terrain + program, powered by CUDA
	*/
	namespace STPCompute {
		__host__ inline void cudaAssert(cudaError_t cuda_code, const char* file, int line);
	}
}

#define STPcudaCheckError(ans) {SuperTerrainPlus::STPCompute::cudaAssert(ans, __FILE__, __LINE__)}

#endif//_STP_DEVICE_ERROR_HANDLER_CUH_