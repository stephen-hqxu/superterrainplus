#pragma once
#ifndef _STP_MEMORY_ERROR_H_
#define _STP_MEMORY_ERROR_H_

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
		 * @brief STPMemoryError tells general error about memory allocation, access and deallocation
		*/
		class STP_API STPMemoryError : public std::logic_error {
		public:

			/**
			 * @brief Init STPMemoryError
			 * @param msg The message about memory error
			*/
			explicit STPMemoryError(const char*);

		};

	}
}
#endif//_STP_MEMORY_ERROR_H_