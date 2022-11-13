#pragma once
#ifndef _STP_VERTEX_ARRAY_H_
#define _STP_VERTEX_ARRAY_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Management
#include "STPNullableObject.hpp"
//Buffer Object
#include "STPBuffer.h"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPVertexArray is a smart wrapper to GL vertex array object.
	*/
	class STP_REALISM_API STPVertexArray {
	public:

		/**
		 * @brief STPVertexAttributeBuilder is a top wrapper to the vertex array for building vertex attributes.
		 * Each call to the the functions will increment the internal counters for offsets therefore allowing fast vertex attributing.
		*/
		class STP_REALISM_API STPVertexAttributeBuilder {
		private:

			const STPOpenGL::STPuint VertexArray;

		public:

			//Accumulative values
			STPOpenGL::STPuint AttribIndex, RelativeOffset;
			//Attribute binding, and the attribute index starting point and ending
			STPOpenGL::STPuint BindingIndex, BlockStart, BlockEnd;

			/**
			 * @brief Init STPVertexAttributeBuilder
			 * @param vao The vertex array object.
			*/
			STPVertexAttributeBuilder(STPOpenGL::STPuint) noexcept;

			~STPVertexAttributeBuilder() = default;

			/**
			 * @brief Specify organisation of vertex arrays.
			 * Vertex attibute index and relative offset is incremented automatically every time this function is called.
			 * @param size The number of values per vertex that are stored in the array. 
			 * @param type The type of the data stored in the array.
			 * @param normalise Specifies whether fixed-point data values should be normalized (GL_TRUE) or 
			 * converted directly as fixed-point values (GL_FALSE) when they are accessed. 
			 * This parameter is ignored if type is GL_FIXED. 
			 * @param attribSize The number of byte this type of format has.
			 * @return The pointer to the current builder for chaining.
			*/
			STPVertexAttributeBuilder& format(STPOpenGL::STPint, STPOpenGL::STPenum, STPOpenGL::STPboolean, unsigned int) noexcept;

			/**
			 * @brief Bind a buffer to the vertex array binding point.
			 * It will bind all previously formatted attribute indices and the current binding block entry.
			 * @param buffer The pointer to the buffer.
			 * @param offset The offset of the first element of the buffer.
			 * @return The pointer to the current builder for chaining.
			*/
			STPVertexAttributeBuilder& vertexBuffer(const STPBuffer&, STPOpenGL::STPintptr) noexcept;

			/**
			 * @brief Bind a element buffer to the current vertex array binding point.
			 * @param buffer The pointer to the element buffer.
			 * @return The pointer to the current builder for chaining.
			*/
			STPVertexAttributeBuilder& elementBuffer(const STPBuffer&) noexcept;

			/**
			 * @brief Bind all previous formatted attribute to the next binding block.
			 * @return The pointer to the current builder for chaining.
			*/
			STPVertexAttributeBuilder& binding() noexcept;

			/**
			 * @brief Set vertex array divisor for the current attribute index.
			 * @param divisor Specify the number of instances that will pass between updates of the generic attribute at slot index. 
			 * @return The pointer to the current builder for chaining.
			*/
			STPVertexAttributeBuilder& divisor(STPOpenGL::STPint) noexcept;

		};

	private:

		/**
		 * @brief STPVertexArrayDeleter is a simple deleter to vertex array.
		*/
		struct STP_REALISM_API STPVertexArrayDeleter {
		public:

			void operator()(STPOpenGL::STPuint) const noexcept;

		};
		typedef STPSmartGLuintObject<STPVertexArrayDeleter> STPSmartVertexArray;
		//VAO
		STPSmartVertexArray VertexArray;

	public:

		/**
		 * @brief Initialise a new vertex array object.
		*/
		STPVertexArray() noexcept;

		STPVertexArray(const STPVertexArray&) = delete;

		STPVertexArray(STPVertexArray&&) noexcept = default;

		STPVertexArray& operator=(const STPVertexArray&) = delete;

		STPVertexArray& operator=(STPVertexArray&&) noexcept = default;

		~STPVertexArray() = default;

		/**
		 * @brief Bind the current vertex array to the GL context.
		*/
		void bind() const noexcept;

		/**
		 * @brief Reset vertex array of the current GL context to zero.
		*/
		static void unbind() noexcept;

		/**
		 * @brief Get the underlying vertex array object.
		 * @return The vertex array.
		*/
		STPOpenGL::STPuint operator*() const noexcept;

		/**
		 * @brief Enable a generic vertex attribute array.
		 * @param index Specifies the index of the generic vertex attribute to be enabled.
		*/
		void enable(STPOpenGL::STPuint) noexcept;

		/**
		 * @brief Enable a vertex attribute array within a range.
		 * @param start The first index to be enabled.
		 * @param count The number of index to be enabled.
		*/
		void enable(STPOpenGL::STPuint, STPOpenGL::STPuint) noexcept;

		/**
		 * @brief Disable a generic vertex attribute array.
		 * @param index Specifies the index of the generic vertex attribute to be disable.
		*/
		void disable(STPOpenGL::STPuint) noexcept;

		/**
		 * @brief Get a vertex array attribute builder.
		 * @return A attribute builder for the current vertex array.
		*/
		STPVertexAttributeBuilder attribute() noexcept;

	};

}
#endif//_STP_VERTEX_ARRAY_H_