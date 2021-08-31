#pragma once
#ifndef _STP_SERIALISATION_ERROR_H_
#define _STP_SERIALISATION_ERROR_H_

#include <STPCoreDefine.h>
//Exception
#include <ios>

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
		 * @brief STPSerialisationError will be thrown when serialisation fails to operate.
		*/
		class STP_API STPSerialisationError : public std::ios_base::failure {
		public:

			/**
			 * @brief Init STPSerialisationError
			 * @param msg Message for serialisation error
			*/
			explicit STPSerialisationError(const char*);

		};

	}
}
#endif //_STP_SERIALISATION_ERROR_H_