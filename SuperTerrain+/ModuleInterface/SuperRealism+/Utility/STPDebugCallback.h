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
		 * @brief Enable GL debug output.
		 * If the current platform does not support this function, exception is thrown.
		*/
		STP_REALISM_API void enable();

		/**
		 * @brief Disable GL debug output.
		*/
		STP_REALISM_API void disable();

		/**
		 * @brief Check if the debug output has been enabled.
		 * @return True if enabled, otherwise false.
		*/
		STP_REALISM_API bool isEnabled();

		/**
		 * @brief Enable asynchronous debug callback with default debug callback function.
		 * The built-in debug callback function writes contents to the output stream.
		 * Debug callback must be enabled first, otherwise exception is thrown.
		 * @param stream The output stream to store the debug message.
		 * The stream must remain valid throughout the life-time of the application until this function is called again with another stream.
		*/
		STP_REALISM_API void enableAsyncCallback(std::ostream&);

		/**
		 * @brief Check if async debug output has been enabled.
		 * Debug callback must first be initialised and enabled.
		 * @return True if enabled.
		*/
		STP_REALISM_API bool isEnabledAsyncCallback();

		/**
		 * @brief Disable asynchronous debug callback.
		*/
		STP_REALISM_API void disableAsyncCallback();

	}

}
#endif//_STP_DEBUG_CALLBACK_H_