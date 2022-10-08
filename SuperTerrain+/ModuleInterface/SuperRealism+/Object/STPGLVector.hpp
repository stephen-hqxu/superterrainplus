#pragma once
#ifndef _STP_GL_VECTOR_HPP_
#define _STP_GL_VECTOR_HPP_

#include <SuperTerrain+/STPOpenGL.h>

//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPGLVector contains some vector types wrapped over some GL types for easier multi-dimension parameter passing.
	*/
	namespace STPGLVector {
		//2-component GLint
		typedef glm::vec<2, STPOpenGL::STPint> STPintVec2;
		//3-component GLint
		typedef glm::vec<3, STPOpenGL::STPint> STPintVec3;
		//4-component GLint
		typedef glm::vec<4, STPOpenGL::STPint> STPintVec4;

		//4-component GLuint
		typedef glm::vec<4, STPOpenGL::STPuint> STPuintVec4;

		//4-component GLfloat
		typedef glm::vec<4, STPOpenGL::STPfloat> STPfloatVec4;

		//2-component GLsizei
		typedef glm::vec<2, STPOpenGL::STPsizei> STPsizeiVec2;
		//3-component GLsizei
		typedef glm::vec<3, STPOpenGL::STPsizei> STPsizeiVec3;
	}

}
#endif//_STP_GL_VECTOR_HPP_