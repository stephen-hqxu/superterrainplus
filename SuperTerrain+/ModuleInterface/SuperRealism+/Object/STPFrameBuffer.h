#pragma once
#ifndef _STP_FRAMEBUFFER_H_
#define _STP_FRAMEBUFFER_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object Management
#include "STPNullableObject.hpp"
//GL Object
#include "STPTexture.h"
#include "STPRenderBuffer.h"

#include <vector>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPFrameBuffer is a high leve wrapper to GL frame buffer object.
	*/
	class STP_REALISM_API STPFrameBuffer {
	private:

		/**
		 * @brief STPFrameBufferDeleter deletes a GL frame buffer object.
		*/
		struct STP_REALISM_API STPFrameBufferDeleter {
		public:

			void operator()(STPOpenGL::STPuint) const;

		};
		typedef STPSmartGLuintObject<STPFrameBufferDeleter> STPSmartFrameBuffer;
		//FBO
		STPSmartFrameBuffer FrameBuffer;

	public:

		/**
		 * @brief Create a new frame buffer object.
		*/
		STPFrameBuffer();

		STPFrameBuffer(const STPFrameBuffer&) = delete;

		STPFrameBuffer(STPFrameBuffer&&) noexcept = default;

		STPFrameBuffer& operator=(const STPFrameBuffer&) = delete;

		STPFrameBuffer& operator=(STPFrameBuffer&&) noexcept = default;

		~STPFrameBuffer() = default;

		/**
		 * @brief Get the GL framebuffer object.
		 * @return The framebuffer object.
		*/
		STPOpenGL::STPuint operator*() const;

		/**
		 * @brief Bind a framebuffer to a framebuffer target.
		 * @param target Specifies the framebuffer target of the binding operation.
		*/
		void bind(STPOpenGL::STPenum) const;

		/**
		 * @brief Break the existing binding of a framebuffer object to target. 
		 * @param target Specifies the framebuffer target of the unbinding operation.
		*/
		static void unbind(STPOpenGL::STPenum);

		/**
		 * @brief Check the completeness status of a framebuffer.
		 * @param target Specify which framebuffer completeness of framebuffer is checked for status.
		 * @return A GL status enum.
		*/
		STPOpenGL::STPenum status(STPOpenGL::STPenum) const;

		/**
		 * @brief Attach a level of a texture object as a logical buffer of a framebuffer object.
		 * @param attachment Specifies the attachment point of the framebuffer.
		 * @param texture The pointer to the managed texture object which specifies the name of an existing texture object to attach.
		 * @param level Specifies the mipmap level of the texture object to attach.
		*/
		void attach(STPOpenGL::STPenum, const STPTexture&, STPOpenGL::STPint);

		/**
		 * @brief Attach a renderbuffer as a logical buffer of a framebuffer object.
		 * @param attachment Specifies the attachment point of the framebuffer.
		 * @param renderbuffer The pointer to the managed render buffer object which 
		 * specifies the name of an existing renderbuffer object of type renderbuffer target to attach.
		*/
		void attach(STPOpenGL::STPenum, const STPRenderBuffer&);

		/**
		 * @brief Detach a texture target from the framebuffer.
		 * @param attachment The attachment on the framebuffer to be detached.
		*/
		void detachTexture(STPOpenGL::STPenum);

		/**
		 * @brief Detach a renderbuffer target from the framebuffer.
		 * @param attachment The attachment on the framebuffer to be detached.
		*/
		void detachRenderBuffer(STPOpenGL::STPenum);

		/**
		 * @brief Specify which colour buffers are to be drawn into.
		 * @param buf The buffer for the framebuffer.
		*/
		void drawBuffer(STPOpenGL::STPenum);

		/**
		 * @brief Specifies a list of colour buffers to be drawn into.
		 * @param bufs Points to an array of symbolic constants specifying the buffers into which fragment colour or data values will be written.
		*/
		void drawBuffers(const std::vector<STPOpenGL::STPenum>&);

		/**
		 * @brief Select a colour buffer source for pixels.
		 * @param mode Specifies a colour buffer.
		*/
		void readBuffer(STPOpenGL::STPenum);

		/**
		 * @brief Clear individual colour buffers of a framebuffer.
		 * @tparam Vec The type of colour vector.
		 * Must be one of the following: uvec4, ivec4 and vec4.
		 * The vector type must corresponds the type of texture attached or it results in undefined behaviour.
		 * @param drawbuffer Specify a particular draw buffer to clear.
		 * @param colour A four-element typed vector specifying the R, G, B and A colour to clear that draw buffer to.
		*/
		void clearColor(STPOpenGL::STPint, const STPGLVector::STPfloatVec4&);
		void clearColor(STPOpenGL::STPint, const STPGLVector::STPintVec4&);
		void clearColor(STPOpenGL::STPint, const STPGLVector::STPuintVec4&);

		/**
		 * @brief Clear depth buffer of a framebuffer.
		 * @param value Value to clear the depth buffer to.
		*/
		void clearDepth(STPOpenGL::STPfloat);

		/**
		 * @brief Clear stencil buffer of a framebuffer.
		 * @param value Value to clear the stencil buffer to.
		*/
		void clearStencil(STPOpenGL::STPint);

		/**
		 * @brief Clear both depth and stencil buffer of a framebuffer.
		 * @param depth The value depth buffer is cleared to.
		 * @param stencil The value stencil buffer is cleared to.
		*/
		void clearDepthStencil(STPOpenGL::STPfloat, STPOpenGL::STPint);

		/**
		 * @brief Copy a block of pixels from one framebuffer object to the current one.
		 * @param readFramebuffer Specifies the name of the source framebuffer object.
		 * @param srcRec Specify the bounds of the source rectangle within the read buffer of the read framebuffer.
		 * @param dstRec Specify the bounds of the destination rectangle within the write buffer of the write framebuffer (the calling framebuffer). 
		 * @param mask The bitwise OR of the flags indicating which buffers are to be copied.
		 * @param filter Specifies the interpolation to be applied if the image is stretched.
		*/
		void blitFrom(const STPFrameBuffer&, const STPGLVector::STPintVec4&, const STPGLVector::STPintVec4&, STPOpenGL::STPbitfield, STPOpenGL::STPenum);

	};

}
#endif//_STP_FRAMEBUFFER_H_