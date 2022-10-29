#pragma once
#ifndef _STP_INIT_H_
#define _STP_INIT_H_

#include <SuperTerrain+/STPCoreDefine.h>

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

		/**
		 * @brief Get the 2-digit GPU architecture representation.
		 * @param device Specifies the device ID to be retrieved.
		 * @return The 2-digit GPU architecture.
		 * For example, compute capability of 7.5 will return 75.
		*/
		STP_API int architecture(int);

	}

}

#endif//_STP_INIT_H_