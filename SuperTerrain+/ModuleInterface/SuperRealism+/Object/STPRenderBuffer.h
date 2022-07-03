#pragma once
#ifndef _STP_RENDERBUFFER_H_
#define _STP_RENDERBUFFER_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object Management
#include "STPNullableObject.hpp"

//GLM
#include <glm/vec2.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPRenderBuffer is smart-managed GL render buffer object.
	*/
	class STP_REALISM_API STPRenderBuffer {
	private:

		/**
		 * @brief STPRenderBufferDeleter deletes a GL render buffer object automatically.
		*/
		struct STP_REALISM_API STPRenderBufferDeleter {
		public:

			void operator()(STPOpenGL::STPuint) const;

		};
		typedef STPSmartGLuintObject<STPRenderBufferDeleter> STPSmartRenderBuffer;
		//RBO
		STPSmartRenderBuffer RenderBuffer;

	public:

		/**
		 * @brief Init a new and empty render buffer object.
		*/
		STPRenderBuffer();

		STPRenderBuffer(const STPRenderBuffer&) = delete;

		STPRenderBuffer(STPRenderBuffer&&) noexcept = default;

		STPRenderBuffer& operator=(const STPRenderBuffer&) = delete;

		STPRenderBuffer& operator=(STPRenderBuffer&&) noexcept = default;

		~STPRenderBuffer() = default;

		/**
		 * @brief Get the underlying managed render buffer object.
		 * @return The render buffer object.
		*/
		STPOpenGL::STPuint operator*() const;

		/**
		 * @brief Bind a renderbuffer to a renderbuffer target.
		*/
		void bind() const;

		/**
		 * @brief Unbind renderbuffer target.
		*/
		static void unbind();

		/**
		 * @brief Establish data storage, format and dimensions of a renderbuffer object's image.
		 * @param internal Specifies the internal format to use for the renderbuffer object's image.
		 * @param dimension The width and height of the renderbuffer, in pixels.
		*/
		void renderbufferStorage(STPOpenGL::STPenum, glm::uvec2);

		/**
		 * @brief Establish data storage, format, dimensions and sample count of a renderbuffer object's image.
		 * @param samples Specifies the number of samples to be used for the renderbuffer object's storage.
		 * @param internal Specifies the internal format to use for the renderbuffer object's image.
		 * @param dimension The width and height of the renderbuffer, in pixels.
		*/
		void renderbufferStorageMultisample(STPOpenGL::STPint, STPOpenGL::STPenum, glm::uvec2);

	};

}
#endif//_STP_RENDERBUFFER_H_