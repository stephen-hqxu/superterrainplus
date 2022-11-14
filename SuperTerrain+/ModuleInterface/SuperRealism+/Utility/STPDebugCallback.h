#pragma once
#ifndef _STP_DEBUG_CALLBACK_H_
#define _STP_DEBUG_CALLBACK_H_

#include <SuperRealism+/STPRealismDefine.h>
#include <SuperTerrain+/STPOpenGL.h>

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
		 * @brief Parse the GL debug callback data and print them to formatted stream output.
		 * The returned message has newline at the end.
		 * @param source The source.
		 * @param type The type.
		 * @param id The ID.
		 * @param severity The severity.
		 * @param length The length.
		 * @param message The message.
		 * @param stream The output stream to store the debug message.
		 * @return The stream same as the input stream.
		*/
		STP_REALISM_API std::ostream& print(STPOpenGL::STPenum, STPOpenGL::STPenum, STPOpenGL::STPuint,
			STPOpenGL::STPenum, STPOpenGL::STPsizei, const char*, std::ostream&);

	}

}
#endif//_STP_DEBUG_CALLBACK_H_