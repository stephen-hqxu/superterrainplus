#pragma once
#ifndef _STP_MATRIX_4X4D_H_
#define _STP_MATRIX_4X4D_H_

#include <SuperTerrain+/STPCoreDefine.h>
#include "STPVector4d.h"

//GLM
#include <glm/mat4x4.hpp>

namespace SuperTerrainPlus {

	/**
	 * @brief STPMatrix4x4d is a 4-by-4 matrix where each component is a double.
	 * It utilises SIMD to improve performance.
	*/
	class STP_API STPMatrix4x4d {
	private:

		STPVector4d Mat[4];

		/**
		 * @brief Get the raw AVX vector from a particular row.
		 * @param idx The row index.
		 * @return The raw vector of the specified row.
		*/
		const __m256d& get(size_t) const noexcept;
		//Similarly but non-const
		__m256d& get(size_t) noexcept;

	public:

		//A 3-by-3 double matrix, which is represented by the 4-by-4 double matrix.
		using STPMatrix3x3d = STPMatrix4x4d;

		/**
		 * @brief Initialise a new STPMatrix4x4d instance with zero value.
		*/
		STPMatrix4x4d() = default;

		/**
		 * @brief Load a new STPMatrix4x4d from a dmat4.
		 * @param mat The matrix to be loaded.
		 * For the best performance, it is recommended that the address is aligned to 32-bit.
		*/
		explicit STPMatrix4x4d(const glm::dmat4&) noexcept;

		STPMatrix4x4d(const STPMatrix4x4d&) = default;

		STPMatrix4x4d& operator=(const STPMatrix4x4d&) = default;

		~STPMatrix4x4d() = default;

		explicit operator glm::dmat4() const noexcept;

		explicit operator glm::mat4() const noexcept;

		/**
		 * @brief Get the row within the matrix.
		 * @param idx The row index.
		 * @return The pointer to the row vector.
		*/
		const STPVector4d& operator[](size_t) const noexcept;
		//Similar, but the non-const version.
		STPVector4d& operator[](size_t) noexcept;

		/**
		 * @brief Multiply with another matrix.
		 * @param rhs The matrix to be multiplied by the current matrix.
		 * @return The resultant matrix after performing operation `this` * `rhs`.
		*/
		STPMatrix4x4d operator*(const STPMatrix4x4d&) const noexcept;

		/**
		 * @brief Multiply with a vector.
		 * @param rhs The vector to be multiplied by the current matrix.
		 * @return The resultant vector after performing the operation `this` * rhs.
		*/
		STPVector4d operator*(const STPVector4d&) const noexcept;

		/**
		 * @brief Transpose the matrix.
		 * @return The transposed matrix.
		*/
		STPMatrix4x4d transpose() const noexcept;

		/**
		 * @brief Invert the matrix.
		 * @return The inverse of the matrix.
		*/
		STPMatrix4x4d inverse() const noexcept;

		/**
		 * @brief Emulate a 3-by-3 double matrix by only preserving the top-left 3-by-3 components.
		 * @return The 4-by-4 matrix emulating a 3-by-3 matrix.
		*/
		STPMatrix3x3d asMatrix3x3d() const noexcept;

	};

}
#endif//_STP_MATRIX_4X4D_H_