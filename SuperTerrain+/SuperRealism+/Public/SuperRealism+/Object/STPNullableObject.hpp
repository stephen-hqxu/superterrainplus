#pragma once
#ifndef _STP_NULLABLE_OBJECT_HPP_
#define _STP_NULLABLE_OBJECT_HPP_

#include <SuperTerrain+/Utility/STPNullablePrimitive.h>
#include <SuperTerrain+/STPOpenGL.h>

namespace SuperTerrainPlus::STPRealism {

	template<class GLuintDel>
	using STPSmartGLuintObject = STPUniqueResource<STPOpenGL::STPuint, 0u, GLuintDel>;

	template<class GLuint64Del>
	using STPSmartGLuint64Object = STPUniqueResource<STPOpenGL::STPuint64, 0ull, GLuint64Del>;

}
#endif//_STP_NULLABLE_OBJECT_HPP_