#pragma once
#ifndef _STP_VECTOR_UTILITY_CUH_
#define _STP_VECTOR_UTILITY_CUH_

//This file provides matrix types extended from CUDA vectors.
//For consistency with OptiX API, all matrices are in row-major.
#ifndef __CUDACC_RTC__
#include <vector_types.h>
#endif

//single precision 4x4 matrix
struct float4x4 {
	float4 x, y, z, w;
};

//memory contingency check
static_assert(sizeof(float4x4) == 16 * sizeof(float), "The current platform uses an incompatible memory layout for structure.");

#ifdef __CUDACC_RTC__
//additional matrix operations for CUDA vector types
#include <sutil/vec_math.h>

SUTIL_INLINE SUTIL_HOSTDEVICE float4 operator*(const float4x4& m, const float4& v) {
	return make_float4(
		dot(m.x, v),
		dot(m.y, v),
		dot(m.z, v),
		dot(m.w, v)
	);
}
#endif//__CUDACC_RTC__
#endif//_STP_VECTOR_UTILITY_CUH_