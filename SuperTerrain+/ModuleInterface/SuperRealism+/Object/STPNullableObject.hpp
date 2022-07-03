#pragma once
#ifndef _STP_NULLABLE_OBJECT_HPP_
#define _STP_NULLABLE_OBJECT_HPP_

#include <SuperTerrain+/Utility/STPNullablePrimitive.h>
#include <SuperTerrain+/STPOpenGL.h>

#include <memory>

namespace SuperTerrainPlus::STPRealism {

	//A nullable GLuint type
	//Most OpenGL objects use GLuint format, doing this allow managing GL objects as a smart pointer.
	typedef STPNullablePrimitive<STPOpenGL::STPuint, 0u> STPNullableGLuint;
	typedef STPNullablePrimitive<STPOpenGL::STPuint64, 0ull> STPNullableGLuint64;

	//The correspond smart object managers
	template<class GLuintDel>
	using STPSmartGLuintObject = std::unique_ptr<STPOpenGL::STPuint, STPNullableGLuint::STPNullableDeleter<GLuintDel>>;

	template<class GLuint64Del>
	using STPSmartGLuint64Object = std::unique_ptr<STPOpenGL::STPuint64, STPNullableGLuint64::STPNullableDeleter<GLuint64Del>>;

}
#endif//_STP_NULLABLE_OBJECT_HPP_