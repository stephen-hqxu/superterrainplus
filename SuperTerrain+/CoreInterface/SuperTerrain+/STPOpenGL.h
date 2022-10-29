#pragma once
#ifndef _STP_OPENGL_H_
#define _STP_OPENGL_H_

#include <cstdint>

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
		//GLsizei
		typedef int STPsizei;
		//GLenum
		typedef unsigned int STPenum;
		//GLbitfield
		typedef unsigned int STPbitfield;
		//GLboolean
		typedef unsigned char STPboolean;
		//GLfloat
		typedef float STPfloat;
		//GLuint64
		typedef std::uint64_t STPuint64;
#ifdef _WIN64
		//GLintptr
		typedef signed long long int STPintptr;
		//GLsizeiptr
		typedef signed long long int STPsizeiptr;
#else
		//GLintptr
		typedef signed long int STPintptr;
		//GLsizeiptr
		typedef signed long int STPsizeiptr;
#endif//_WIN64
	}

}

#endif//_STP_OPENGL_H_