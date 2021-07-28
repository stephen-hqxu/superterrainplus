#pragma once
#ifndef _STP_INIT_H_
#define _STP_INIT_H_

#include <STPCoreDefine.h>

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPEngineInitialiser initialises SuperTerrain+ main engine.
	 * Making any function call before engine is initialised will result in underfined behaviour
	*/
	class STP_API STPEngineInitialiser {
	private:

		//Default state is false, once the engine is initialised it will become true.
		static bool Inited;

		/**
		 * @brief A static-only class should not be constructed
		*/
		STPEngineInitialiser() = default;

		~STPEngineInitialiser() = default;

	public:

		//Indicate the process with OpenGL context
		typedef void* (*STPglProc)(const char* name);

		/**
		 * @brief Initialise OpenGL context with the current process.
		 * @return True if OpenGL init is successful
		*/
		static bool initGLcurrent();

		/**
		 * @brief Initialise OpenGL context with explictly specified process
		 * @param process The process to use
		 * @return True if OpenGL init is successful
		*/
		static bool initGLexplicit(STPglProc);

		/**
		 * @brief Check if the engine has been initialised
		 * @return True if engien has been initialised completely.
		*/
		static bool hasInit();

	};

}

#endif//_STP_INIT_H_