#pragma once
#ifndef _STP_VECTOR_4D_H_
#define _STP_VECTOR_4D_H_

#include "STPAlgebraCommon.h"

//GLM
#include <glm/vec4.hpp>

namespace SuperTerrainPlus {

	/**
	 * @brief STPVector4d is a 4-component vector class where each component is a double.
	 * It utilises SIMD instructions to improve performance.
	*/
	class STPVector4d {
	public:

		/**
		 * @brief STPElement can be used to locate the component under the vector.
		*/
		enum class STPElement : unsigned char {
			X = 0x00u,
			Y = 0x01u,
			Z = 0x02u,
			W = 0x03u
		};

	private:

		friend class STPMatrix4x4d;

		__m256d Vec;

		/**
		 * @brief Initialise by the raw AVX vector.
		 * @param vec The value of the AVX vector.
		*/
		STPVector4d(const __m256d&) noexcept;

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

		STPVector4d(const STPVector4d&) = default;

		STPVector4d& operator=(const STPVector4d&) = default;

		~STPVector4d() = default;

		explicit operator glm::dvec4() const noexcept;

		explicit operator glm::vec4() const noexcept;

		/**
		 * @brief Perform addition on the target vector.
		 * @param rhs The vector value.
		 * @return The result of `this` + `rhs`.
		*/
		STPVector4d operator+(const STPVector4d&) const noexcept;

		/**
		 * @brief Perform division on the target vector.
		 * @param rhs The vector value.
		 * @return The result of `this` / `rhs`.
		*/
		STPVector4d operator/(const STPVector4d&) const noexcept;

		/**
		 * @brief Broadcast one element in the vector to the entire vector.
		 * @tparam E Specify the component of the element to be broadcast.
		 * @return The vector where each element is the broadcast of one element in the original vector.
		*/
		template<STPElement E>
		STPVector4d broadcast() const noexcept;

		/**
		 * @brief Perform vector dot product operation.
		 * @param rhs The vector value.
		 * @return The result of `this` dot `rhs`.
		*/
		double dot(const STPVector4d&) const noexcept;

	};

}
#include "STPVector4d.inl"
#endif//_STP_VECTOR_4D_H_