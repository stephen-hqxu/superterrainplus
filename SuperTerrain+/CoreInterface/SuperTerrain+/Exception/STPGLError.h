#pragma once
#ifndef _STP_GL_ERROR_H_
#define _STP_GL_ERROR_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Exception
#include <stdexcept>

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPGLError specifies error thrown from OpenGL API.
	*/
	class STP_API STPGLError : public std::runtime_error {
	public:

		/**
		 * @brief Init STPGLError
		 * @param msg The OpenGL error message
		*/
		explicit STPGLError(const char*);

	};

}
#endif//_STP_GL_ERROR_H_