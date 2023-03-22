#pragma once
#ifndef _STP_BUFFER_H_
#define _STP_BUFFER_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object Management
#include "STPNullableObject.hpp"

//Container
#include <vector>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPBuffer is a high level wrapper to GL buffer object.
	*/
	class STP_REALISM_API STPBuffer {
	private:

		/**
		 * @brief STPBufferDeleter deletes GL buffer
		*/
		struct STP_REALISM_API STPBufferDeleter {
		public:

			void operator()(STPOpenGL::STPuint) const noexcept;

		};
		typedef STPSmartGLuintObject<STPBufferDeleter> STPSmartBuffer;
		//VBO
		STPSmartBuffer Buffer;

	public:

		/**
		 * @brief Initialise an empty GL buffer object.
		*/
		STPBuffer() noexcept;

		STPBuffer(const STPBuffer&) = delete;

		STPBuffer(STPBuffer&&) noexcept = default;

		STPBuffer& operator=(const STPBuffer&) = delete;

		STPBuffer& operator=(STPBuffer&&) noexcept = default;

		~STPBuffer() = default;

		/**
		 * @brief Get the GL buffer object managed.
		 * @return The GL buffer object.
		*/
		STPOpenGL::STPuint operator*() const noexcept;

		/**
		 * @brief Bind the current GL buffer to a target in the context.
		 * @param target The GL target being bound to.
		*/
		void bind(STPOpenGL::STPenum) const noexcept;

		/**
		 * @brief Bind a buffer object to an indexed buffer target.
		 * @param target Specify the target of the bind operation.
		 * @param index Specify the index of the binding point within the array specified by target.
		*/
		void bindBase(STPOpenGL::STPenum, STPOpenGL::STPuint) const noexcept;

		/**
		 * @brief Unbind from an indexed buffer target.
		 * @param target The targe to be unbound.
		 * @param index The index to be unbound.
		*/
		static void unbindBase(STPOpenGL::STPenum, STPOpenGL::STPuint) noexcept;

		/**
		 * @brief Query the GPU address of the current buffer.
		 * This address remains valid until the buffer object is deleted or reallocated.
		 * Bindless buffer is a functionality provided by NV_shader_buffer_load and/or NV_shader_buffer_store.
		 * @return The GPU address of the current buffer.
		*/
		STPOpenGL::STPuint64 getAddress() const noexcept;

		/**
		 * @brief The data store of a buffer object may be made accessible to the GL via buffer GPU address by calling this function.
		 * @param access Specifies the memory read write mode with the buffer.
		*/
		void makeResident(STPOpenGL::STPenum) const noexcept;

		/**
		 * @brief The data store of a buffer object may be made inaccessible to the GL via buffer address by calling this function.
		*/
		void makeNonResident() const noexcept;

		/**
		 * @brief Map all of a buffer object's data store into the client's address space.
		 * @param access The flag for access policies.
		 * @return A pointer to the mapped buffer.
		*/
		void* mapBuffer(STPOpenGL::STPenum) noexcept;

		/**
		 * @brief Map all or part of a buffer object's data store into the client's address space.
		 * @param offset Specifies the starting offset within the buffer of the range to be mapped.
		 * @param length Specifies the length of the range to be mapped.
		 * @param access Specifies a combination of access flags indicating the desired access to the mapped range.
		 * @return A pointer to the mapped buffer.
		*/
		void* mapBufferRange(STPOpenGL::STPintptr, STPOpenGL::STPsizeiptr, STPOpenGL::STPbitfield) noexcept;

		/**
		 * @brief Indicate modifications to a range of a mapped buffer.
		 * @param offset Specifies the start of the buffer subrange, in basic machine units.
		 * @param length Specifies the length of the buffer subrange, in basic machine units.
		*/
		void flushMappedBufferRange(STPOpenGL::STPintptr, STPOpenGL::STPsizeiptr) noexcept;

		/**
		 * @brief Release the mapping of a buffer object's data store into the client's address space.
		 * @return Generally it returns true.
		 * Unless the data store contents have become corrupt during the time the data store was mapped.
		*/
		STPOpenGL::STPboolean unmapBuffer() const noexcept;

		/**
		 * @brief Unbind GL buffer from the current context.
		 * @param target The GL target to be unbound.
		*/
		static void unbind(STPOpenGL::STPenum) noexcept;

		/**
		 * @brief Allocate immutable storage with no data.
		 * @param size The number of byte to be allocated.
		 * @param flag Flags for immutable storage.
		*/
		void bufferStorage(STPOpenGL::STPsizeiptr, STPOpenGL::STPbitfield) noexcept;

		/**
		 * @brief Transfer data to the storage.
		 * @param data An array of data to be transferred.
		 * @param size The number of byte to be passed to the storage.
		 * @param offset Specifies the offset into the buffer object's data store where data replacement will begin, measured in bytes. 
		*/
		void bufferSubData(const void*, STPOpenGL::STPsizeiptr, STPOpenGL::STPintptr) noexcept;

		/**
		 * @brief Allocate immutable storage for a buffer and transfer data to the storage.
		 * @param data An array of data to be submitted.
		 * @param size The number of byte to be passed to the storage.
		 * @param flag Flags for immutable storage.
		*/
		void bufferStorageSubData(const void*, STPOpenGL::STPsizeiptr, STPOpenGL::STPbitfield) noexcept;

		/**
		 * @brief Copy all or part of the data store of a buffer object to the current buffer object.
		 * @param readBuffer Specifies the source buffer object.
		 * @param readOffset Specifies the offset, in basic machine units, within the data store of the source buffer object at which data will be read.
		 * @param writeOffset Specifies the offset, in basic machine units, within the data store of the destination buffer object at which data will be written.
		 * @param size Specifies the size, in basic machine units, of the data to be copied from the source buffer object to the destination buffer object.
		*/
		void copyBufferSubDataFrom(const STPBuffer&, STPOpenGL::STPintptr, STPOpenGL::STPintptr, STPOpenGL::STPsizeiptr) noexcept;

	};

}
#endif//_STP_BUFFER_H_