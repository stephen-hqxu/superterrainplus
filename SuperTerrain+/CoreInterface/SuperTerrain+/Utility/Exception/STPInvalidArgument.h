#pragma once
#ifndef _STP_INVALID_ARGUMENT_H_
#define _STP_INVALID_ARGUMENT_H_

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
		 * @brief STPInvalidArgument states the function parameter is not valid for the function call
		*/
		class STP_API STPInvalidArgument : public std::invalid_argument {
		public:

			/**
			 * @brief Init STPInvalidArgument
			 * @param msg Meesage about the invalid argument
			*/
			explicit STPInvalidArgument(const char*);

		};

	}
}
#endif//_STP_INVALID_ARGUMENT_H_