#pragma once
#ifndef _STP_DEBUG_CALLBACK_H_
#define _STP_DEBUG_CALLBACK_H_

#include <SuperRealism+/STPRealismDefine.h>

//Stream
#include <ostream>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPDebugCallback is a simple wrapper to GL debug callback.
	 * This functionality requires support for ARB_debug_output, as a core module since OpenGL 4.3.
	 * @see https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_debug_output.txt
	*/
	namespace STPDebugCallback {

		/**
		 * @brief Check if the graphics card support debug callback.
		 * @return True if GPU has support, otherwise false.
		*/
		STP_REALISM_API int support();

		/**
		 * @brief Enable asynchronous debug callback with default debug callback function.
		 * The built-in debug callback function writes contents to the output stream.
		 * Debug callback must be enabled first.
		 * @param stream The output stream to store the debug message.
		 * The stream must remain valid throughout the life-time of the application until this function is called again with another stream.
		*/
		STP_REALISM_API void registerAsyncCallback(std::ostream&);

	}

}
#endif//_STP_DEBUG_CALLBACK_H_