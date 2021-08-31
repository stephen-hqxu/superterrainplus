#pragma once
#ifndef _STP_UNSUPPORTED_FUNCTIONALITY_H_
#define _STP_UNSUPPORTED_FUNCTIONALITY_H_

#include <STPCoreDefine.h>
//Exception
#include "STPInvalidArgument.h"

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
		 * @brief STPUnsupportedFunctionality states the functionality that user is asking for is not (yet) supported or valid.
		*/
		class STP_API STPUnsupportedFunctionality : public STPInvalidArgument {
		public:

			/**
			 * @brief Init STPUnsupportedFunctionality
			 * @param msg The message for what functionality is not supported
			*/
			explicit STPUnsupportedFunctionality(const char*);

		};

	}
}
#endif//_STP_UNSUPPORTED_FUNCTIONALITY_H_