#pragma once
#ifndef _STP_VECTOR_4D_H_
#define _STP_VECTOR_4D_H_

#include <SuperTerrain+/STPCoreDefine.h>

#include <immintrin.h>

//GLM
#include <glm/vec4.hpp>

namespace SuperTerrainPlus {

	/**
	 * @brief STPVector4d is a 4-component vector class where each component is a double.
	 * It utilises SIMD instructions to improve performance.
	*/
	class STP_API STPVector4d {
	private:

		friend class STPMatrix4x4d;

		__m256d Vec;

		/**
		 * @brief Perform dot product on the raw AVX vectors.
		 * @param lhs The LHS vector.
		 * @param rhs The RHS vector.
		 * @return The resultant AVX vector of `lhs` dot `rhs`.
		 * All components have the same value of the dot product result.
		*/
		static __m256d dotVector4dRaw(const __m256d&, const __m256d&) noexcept;

	public:

		/**
		 * @brief Initialise a new STPVector4d instance with zero value.
		*/
		STPVector4d() noexcept;

		/**
		 * @brief Load a new STPVector4d instance from a dvec4.
		 * @param vec The vector data to load from.
		 * For the best performance, it is recommended that the address is aligned to 32-bit.
		*/
		explicit STPVector4d(const glm::dvec4&) noexcept;

		~STPVector4d() = default;

		explicit operator glm::dvec4() const noexcept;

		/**
		 * @brief Perform vector dot product operation.
		 * @param rhs The vector value.
		 * @return The result of `this` dot `rhs`.
		*/
		double dot(const STPVector4d&) const noexcept;

	};

}
#endif//_STP_VECTOR_4D_H_