#pragma once
#ifndef _STP_CUDA_ERROR_H_
#define _STP_CUDA_ERROR_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Exception
#include <stdexcept>

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {
	/**
	 * @brief STPException provides a variety of exception classes for Super Terrain + engine.
	*/
	namespace STPException {

		/**
		 * @brief STPCUDAError specifies error thrown from CUDA API.
		*/
		class STP_API STPCUDAError : public std::runtime_error {
		public:

			/**
			 * @brief Init STPCUDAError
			 * @param msg The CUDA error message
			*/
			explicit STPCUDAError(const char*);

		};

	}
}

#endif//_STP_CUDA_ERROR_H_