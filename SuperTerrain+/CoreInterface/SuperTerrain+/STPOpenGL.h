#pragma once
#ifndef _STP_OPENGL_H_
#define _STP_OPENGL_H_

#ifndef __CUDACC_RTC__
//NVRTC does not include any system header, so we need to exclude it
//CUDA does not use GL stuff anyway.
#include <cstdint>
#endif//__CUDACC_RTC__

namespace SuperTerrainPlus {
	
	/**
	 * @brief STPOpenGL is a compatibility header to OpenGL types.
	 * It can be used in library header to represent some GL types so client doesn't have to include GL library (if they don't want to)
	*/
	namespace STPOpenGL {
		//GLuint
		typedef unsigned int STPuint;
		//GLint
		typedef int STPint;
		//GLenum
		typedef unsigned int STPenum;
		//GLbitfield
		typedef unsigned int STPbitfield;
		//GLboolean
		typedef unsigned char STPboolean;
		//GLfloat
		typedef float STPfloat;
#ifndef __CUDACC_RTC__
		//GLuint64
		typedef uint64_t STPuint64;
		//GLintptr
		typedef intptr_t STPintptr;
#endif//__CUDACC_RTC__
	}

}

#endif//_STP_OPENGL_H_