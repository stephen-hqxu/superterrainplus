#pragma once
#ifndef _STP_VECTOR_UTILITY_CUH_
#define _STP_VECTOR_UTILITY_CUH_

//extended CUDA vector maths
#include <sutil/vec_math.h>

//matrix types following CUDA convention
//for consistency with OptiX API, all matrices are in row-major
struct float4x4 {
	float4 x, y, z, w;
};

//additional matrix operations
SUTIL_INLINE SUTIL_HOSTDEVICE float4 operator*(const float4x4& m, const float4& v) {
	return make_float4(
		dot(m.x, v),
		dot(m.y, v),
		dot(m.z, v),
		dot(m.w, v)
	);
}

#endif//_STP_VECTOR_UTILITY_CUH_