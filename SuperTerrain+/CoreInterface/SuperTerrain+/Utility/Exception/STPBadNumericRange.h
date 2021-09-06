#pragma once
#ifndef _STP_BAD_NUMERIC_RANGE_H_
#define _STP_BAD_NUMERIC_RANGE_H_

#include <SuperTerrain+/STPCoreDefine.h>
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
		 * @brief STPBadNumericRange defines number provided is out-of legal range and will cause fatal error if proceed
		*/
		class STP_API STPBadNumericRange : public STPInvalidArgument {
		public:

			/**
			 * @brief Init STPBadNumericRange
			 * @param msg The message about the numeric error
			*/
			explicit STPBadNumericRange(const char*);

		};
	}
}
#endif//_STP_BAD_NUMERIC_RANGE_H_