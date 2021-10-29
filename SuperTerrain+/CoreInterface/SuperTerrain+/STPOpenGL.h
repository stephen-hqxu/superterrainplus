#pragma once
#ifndef _STP_OPENGL_H_
#define _STP_OPENGL_H_

namespace SuperTerrainPlus {
	
	/**
	 * @brief STPOpenGL is a compatibility header to OpenGL types.
	 * It can be used in library header to represent some GL types so client doesn't have to include GL library (if they don't want to)
	*/
	namespace STPOpenGL {
		//GLuint
		typedef unsigned int STPuint;
		//GLenum
		typedef unsigned int STPenum;
	}

}

#endif//_STP_OPENGL_H_