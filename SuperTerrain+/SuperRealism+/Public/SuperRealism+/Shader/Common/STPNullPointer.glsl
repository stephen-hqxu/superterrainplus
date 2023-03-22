#ifndef _STP_NULL_POINTER_GLSL_
#define _STP_NULL_POINTER_GLSL_

//Check if a pointer is null.
//To use pointer in a shading language, certain extensions are required.
bool isNull(const void* const ptr) {
	return unpackPtr(ptr) == uvec2(0u);
}

#endif//_STP_NULL_POINTER_GLSL_