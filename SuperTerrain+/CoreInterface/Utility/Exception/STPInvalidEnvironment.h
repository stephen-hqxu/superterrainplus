#pragma once
#ifndef _STP_INVALID_ENVIRONMENT_H_
#define _STP_INVALID_ENVIRONMENT_H_

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
		 * @brief STPInvalidEnvironment specifies the values from STPEnvironment are not validated
		*/
		class STP_API STPInvalidEnvironment : public STPInvalidArgument {
		public:

			/**
			 * @brief Init STPInvalidEnvironment
			 * @param msg The mssage about the invalid environment
			*/
			explicit STPInvalidEnvironment(const char*);

		};

	}
}
#endif//_STP_INVALID_ENVIRONMENT_H_