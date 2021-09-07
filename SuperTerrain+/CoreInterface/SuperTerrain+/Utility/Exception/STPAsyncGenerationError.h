#pragma once
#ifndef _STP_ASYNC_GENERATION_ERROR_H_
#define _STP_ASYNC_GENERATION_ERROR_H_

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
		 * @brief STPAsyncGenerationError indicates some exception is thrown during async launch to compute the terrain.
		*/
		class STP_API STPAsyncGenerationError : std::runtime_error {
		public:

			/**
			 * @brief Init STPAsyncGenerationError
			 * @param msg The mssage about the async failure
			*/
			explicit STPAsyncGenerationError(const char*);

		};

	}
}
#endif//_STP_ASYNC_GENERATION_ERROR_H_