#pragma once
#ifndef _STP_NULLABLE_PRIMITIVE_H_
#define _STP_NULLABLE_PRIMITIVE_H_

#include <memory>

namespace SuperTerrainPlus {

	/**
	 * @brief STPNullablePrimitive allows hacking a primitive value into a fake pointer.
	 * This can be beneficial to manage resource as a unique_ptr without allocating dynamic memory.
	 * It satisfies NullablePointer requirement.
	 * @tparam Pri A primitive type, or any type that allows null comparison.
	 * For the best performance, keep this type small (no more than size of a pointer is desired).
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
		Pri Value;

	public:

		/**
		 * @brief Initialise a nullable primitive with value of Null (nullptr)
		 * @param A null pointer.
		*/
		STPNullablePrimitive(std::nullptr_t = nullptr) noexcept;

		/**
		 * @brief Initialise a nullable primitive with a value.
		 * @param value The value of this nullable primitive.
		*/
		STPNullablePrimitive(Pri) noexcept;

		/**
		 * @brief Make the nullable primitive null.
		 * @param A null pointer.
		 * @return The primitive that has value equivalent to its null.
		*/
		STPNullablePrimitive& operator=(std::nullptr_t) noexcept;

		/**
		 * @brief Convert the nullable primitive to the actual primitive.
		*/
		operator Pri() const noexcept;

		/**
		 * @brief Convert the nullable primitive to bool, based on NullablePointer specification.
		 * It is contextually convertible to bool.
		 * @return True if object is not null, false otherwise.
		*/
		explicit operator bool() const noexcept;

		/* comparator */

		bool operator==(STPNullablePrimitive) const noexcept;
		bool operator!=(STPNullablePrimitive) const noexcept;

		bool operator==(std::nullptr_t) const noexcept;
		bool operator!=(std::nullptr_t) const noexcept;

	};

	/**
	 * @brief STPUniqueResource is a thin wrapper over std::unique_ptr that allows automatic lifetime management for an arbitrary type of resource
	 * rather than raw pointers only.
	 * @tparam Pri The type of the primitive resource.
	 * @param Null The value denoting a null for this type.
	 * @tparam Del The deleter function to this type of resource.
	*/
	template<class Pri, Pri Null, class Del>
	using STPUniqueResource = std::unique_ptr<Pri, typename STPNullablePrimitive<Pri, Null>::template STPNullableDeleter<Del>>;

}
#include "STPNullablePrimitive.inl"
#endif//_STP_NULLABLE_PRIMITIVE_H_