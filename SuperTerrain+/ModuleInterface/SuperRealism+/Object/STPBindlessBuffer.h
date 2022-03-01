#pragma once
#ifndef _STP_BINDLESS_BUFFER_H_
#define _STP_BINDLESS_BUFFER_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object
#include "STPBuffer.h"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPBindlessBuffer allows getting a GPU address of a OpenGL buffer object and use this address in the shader to eliminate binding.
	 * Bindless buffer is a functionality provided by NV_shader_buffer_load.
	 * Associated GL buffer object must remain valid until the bindless buffer instance is destroyed.
	*/
	class STP_REALISM_API STPBindlessBuffer {
	private:

		/**
		 * @brief STPBindlessBufferInvalidater automatically unresident a bindless buffer address.
		*/
		struct STP_REALISM_API STPBindlessBufferInvalidater {
		public:

			void operator()(STPOpenGL::STPuint) const;

		};
		typedef std::unique_ptr<STPOpenGL::STPuint, STPNullableGLuint::STPNullableDeleter<STPBindlessBufferInvalidater>> STPSmartBindlessBuffer;
		//Bindless VBO
		STPSmartBindlessBuffer Buffer;
		//The address aqured from the buffer.
		STPOpenGL::STPuint64 Address;

	public:

		/**
		 * @brief Create a bindless buffer from a buffer object.
		 * @param buffer The pointer to the buffer object for retrieving address.
		 * @param access Specifies the memory access method for this buffer address.
		*/
		STPBindlessBuffer(const STPBuffer&, STPOpenGL::STPenum);

		STPBindlessBuffer(const STPBindlessBuffer&) = delete;

		STPBindlessBuffer(STPBindlessBuffer&&) noexcept = default;

		STPBindlessBuffer& operator=(const STPBindlessBuffer&) = delete;

		STPBindlessBuffer& operator=(STPBindlessBuffer&&) noexcept = default;

		~STPBindlessBuffer() = default;

		/**
		 * @brief Get the bindless buffer address.
		 * @return The address to the bindless buffer.
		*/
		STPOpenGL::STPuint64 operator*() const;

	};

}
#endif//_STP_BINDLESS_BUFFER_H_