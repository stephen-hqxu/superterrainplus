#pragma once
#ifndef _STP_INIT_H_
#define _STP_INIT_H_

#include <SuperTerrain+/STPCoreDefine.h>

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
	namespace STPEngineInitialiser {

		//Indicate the process with OpenGL context
		typedef void* (*STPglProc)(const char* name);

		/**
		 * @brief Initialise OpenGL context with the current process.
		 * @return True if OpenGL init is successful
		*/
		STP_API bool initGLcurrent();

		/**
		 * @brief Initialise OpenGL context with explictly specified process
		 * @param process The process to use
		 * @return True if OpenGL init is successful
		*/
		STP_API bool initGLexplicit(STPglProc);

		/**
		 * @brief Initialise super terrain plus engine
		 * @param device Specify which CUDA-enabled GPU will be used for computing
		*/
		STP_API void init(int);

		/**
		 * @brief Check if the engine has been initialised
		 * @return True if engien has been initialised completely.
		*/
		STP_API bool hasInit();

	};

}

#endif//_STP_INIT_H_