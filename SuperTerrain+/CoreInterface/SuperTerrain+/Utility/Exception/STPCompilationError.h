#pragma once
#ifndef _STP_COMPILATION_ERROR_H_
#define _STP_COMPILATION_ERROR_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Exception
#include "STPCUDAError.h"

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
		 * @brief STPCompilationError indicates compilation failure during runtime compilation of CUDA code.
		*/
		class STP_API STPCompilationError : public STPCUDAError {
		public:

			/**
			 * @brief Init STPCompilationError
			 * @param msg The error message from CUDA runtime compiler.
			*/
			explicit STPCompilationError(const char*);

		};

	}
}
#endif//_STP_COMPILATION_ERROR_H_