#pragma once
#ifndef _STP_NULLABLE_PRIMITIVE_H_
#define _STP_NULLABLE_PRIMITIVE_H_

namespace SuperTerrainPlus {

	/**
	 * @brief STPNullablePrimitive allows hacking a primitive value into a fake pointer.
	 * This can be beneficial to manage resource as a unique_ptr without allocating dynamic memory.
	 * @tparam Pri A primitive type, or any type that allows null comparison.
	 * @param Null Provide a value that denotes Null for this type.
	*/
	template<typename Pri, Pri Null>
	class STPNullablePrimitive {
	public:

		/**
		 * @brief STPNullableDeleter is a simple wrapper to a nullable primitive for deletion.
		 * @tparam Del The deleter for this primitive.
		*/
		template<class Del>
		struct STPNullableDeleter {
		private:

			Del Deleter;

		public:

			using pointer = STPNullablePrimitive;

			/**
			 * @brief Delete the nullable primitive.
			 * @param ptr The "pointer" to the primitive value.
			*/
			void operator()(pointer) const noexcept;

		};

	private:

		//The underlying primitive value
		Pri Value = Null;

	public:

		/**
		 * @brief Initialise a nullable primitive with default value
		*/
		STPNullablePrimitive() noexcept = default;

		/**
		 * @brief Initialise a nullable primitive with value of Null (nullptr)
		 * @param A null pointer.
		*/
		STPNullablePrimitive(std::nullptr_t) noexcept;

		/**
		 * @brief Initialise a nullable primitive with a value.
		 * @param value The value of this nullable primitive.
		*/
		STPNullablePrimitive(Pri) noexcept;

		/**
		 * @brief Convert the nullable primitive to the actual primitive.
		*/
		operator Pri() const noexcept;

		/**
		 * @brief Convert the nullable primitive to the actual primitive.
		 * @return The primitive value;
		*/
		Pri operator*() const noexcept;

		/* Nullable comparator */

		bool operator==(std::nullptr_t) const noexcept;
		bool operator!=(std::nullptr_t) const noexcept;

	};

}
#include "STPNullablePrimitive.inl"
#endif//_STP_NULLABLE_PRIMITIVE_H_