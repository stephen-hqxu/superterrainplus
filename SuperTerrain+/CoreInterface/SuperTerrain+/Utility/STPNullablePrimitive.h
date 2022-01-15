#pragma once
#ifndef _STP_NULLABLE_PRIMITIVE_H_
#define _STP_NULLABLE_PRIMITIVE_H_

#include <SuperTerrain+/STPCoreDefine.h>
//System
#include <memory>
//GL
#include <SuperTerrain+/STPOpenGL.h>

namespace SuperTerrainPlus {

	/**
	 * @brief STPNullablePrimitive allows hacking a primitive value into a fake pointer.
	 * This can be benefitial to manage resource as a unique_ptr without allocating dynamic memory.
	 * @tparam Pri A primitive type, or any type that allows null comparison.
	 * @param Null Provide a value that denotes Null for this type.
	*/
	template<class Pri, Pri Null>
	struct STP_API STPNullablePrimitive {
	public:

		/**
		 * @brief STPNullableDeleter is a simple wrapper to a nullable primitive for deletion.
		 * @tparam Del The deleter for this primitive.
		*/
		template<class Del>
		struct STPNullableDeleter {
		private:

			Del deleter;

		public:

			using pointer = STPNullablePrimitive;

			/**
			 * @brief Delete the nullable primitive.
			 * @param ptr The "pointer" to the primitive value.
			*/
			void operator()(pointer ptr) const {
				this->deleter(ptr);
			}

		};

	private:

		//The underlying primitive value
		Pri Value = Null;

	public:

		/**
		 * @brief Intialise a nullable primitive with default value
		*/
		STPNullablePrimitive() = default;

		/**
		 * @brief Initialise a nullable primitive with value of Null (nullptr)
		 * @param A null pointer.
		*/
		STPNullablePrimitive(std::nullptr_t);

		/**
		 * @brief Initialse a nullable primitive with a value.
		 * @param value The value of this nullable primitive.
		*/
		STPNullablePrimitive(Pri);

		/**
		 * @brief Convert the nullable primitive to the actual primitive.
		*/
		operator Pri() const;

		/**
		 * @brief Convert the nullable primitive to the actual primitive.
		 * @return The primitive value;
		*/
		Pri operator*() const;

		/* Nullable comparator */

		bool operator==(std::nullptr_t) const;
		bool operator!=(std::nullptr_t) const;

	};

	//A nullable GLuint type
	//Most OpenGL objects use GLuint format, doing this allow manging GL objects as a smart pointer.
	typedef STPNullablePrimitive<STPOpenGL::STPuint, 0u> STPNullableGLuint;
	typedef STPNullablePrimitive<STPOpenGL::STPuint64, 0ull> STPNullableGLuint64;

}
#endif//_STP_NULLABLE_PRIMITIVE_H_