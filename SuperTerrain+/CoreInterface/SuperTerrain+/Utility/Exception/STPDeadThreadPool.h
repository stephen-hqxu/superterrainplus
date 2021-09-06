#pragma once
#ifndef _STP_DEAD_THREAD_POOL_H_
#define _STP_DEAD_THREAD_POOL_H_

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
		 * @brief STPDeadThreadPool is the error thrown when enqueuing a new task to a thread pool that is not running
		*/
		class STP_API STPDeadThreadPool : public std::runtime_error {
		public:

			/**
			 * @brief Init STPDeadThreadPool
			 * @param msg The message to inform user about the dead thread pool
			*/
			explicit STPDeadThreadPool(const char*);

		};

	}
}
#endif//_STP_DEAD_THREAD_POOL_H_