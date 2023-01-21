#pragma once
#ifndef _STP_ALGEBRA_COMMON_H_
#define _STP_ALGEBRA_COMMON_H_

#ifdef __AVX__
//Indication of availability of SIMD instruction
#define STP_ARCH_SIMD
#endif

#ifdef STP_ARCH_SIMD
#include <immintrin.h>

#define STP_MM_BIT4(D, C, B, A) (A << 3 | B << 2 | C << 1 | D)
#define STP_MM_BIT8(D, C, B, A) _MM_SHUFFLE(A, B, C, D)
#endif

#include <cstdint>

namespace SuperTerrainPlus {

	/**
	 * @brief STPAlgebra is a collection of linear algebra library for computer graphics.
	 * The implementation of all SIMD algebras reference the implementation by GLM.
	 * GLM SIMD library does not support double type, I adapted their solution for double precision data type.
	 * Documentations are mostly taken from their code.
	 * 
	 * It is much harder to implement SIMD for double due to lane size (128-bit) and cross-lane operations.
	 * I have tried my best to use the fewest instructions with the minimal instruction latency and CPI to achieve the same functionality.
	*/
	namespace STPAlgebraCommon {

		//The default alignment of the SSE and AVX instruction set
		inline constexpr std::uintptr_t SSEAlignment = alignof(__m128),
			AVXAlignment = alignof(__m256d);

		/**
		 * @brief Check if the address is properly aligned such that it satisfies the alignment requirement of AVX instruction set.
		 * @param addr The address to be checked.
		 * @return True if the address is properly aligned, false otherwise.
		*/
		bool isAVXAligned(const void*) noexcept;

	}

}
#include "STPAlgebraCommon.inl"
#endif//_STP_ALGEBRA_COMMON_H_