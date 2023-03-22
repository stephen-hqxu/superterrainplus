#pragma once
#ifndef _STP_INIT_H_
#define _STP_INIT_H_

#include <SuperTerrain+/STPCoreDefine.h>

namespace SuperTerrainPlus {

	/**
	 * @brief STPEngineInitialiser initialises SuperTerrain+ main engine.
	 * Making any function call before engine is initialised will result in undefined behaviour
	*/
	namespace STPEngineInitialiser {

		//Indicate the process with OpenGL context
		typedef void* (*STPGLProc)(const char* name);

		/**
		 * @brief Initialise the engine.
		 * @param device Specify the device where CUDA works should be carried on.
		 * @param gl_process Specify the process handle to initialise the GL context.
		 * If it is a nullptr, it will skip GL context initialisation (so no context is created).
		*/
		STP_API void initialise(int, STPGLProc);

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