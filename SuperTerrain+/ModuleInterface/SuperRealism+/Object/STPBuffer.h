#pragma once
#ifndef _STP_BUFFER_H_
#define _STP_BUFFER_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object Management
#include <SuperTerrain+/Utility/STPNullablePrimitive.h>

//Container
#include <vector>
#include <memory>

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

			void operator()(STPOpenGL::STPuint) const;

		};
		typedef std::unique_ptr<STPOpenGL::STPuint, STPNullableGLuint::STPNullableDeleter<STPBufferDeleter>> STPSmartBuffer;
		//VBO
		STPSmartBuffer Buffer;

	public:

		/**
		 * @brief Initialise an empty GL buffer object.
		*/
		STPBuffer();

		STPBuffer(const STPBuffer&) = delete;

		STPBuffer(STPBuffer&&) noexcept = default;

		STPBuffer& operator=(const STPBuffer&) = delete;

		STPBuffer& operator=(STPBuffer&&) noexcept = default;

		~STPBuffer() = default;

		/**
		 * @brief Get the GL buffer object managed.
		 * @return The GL buffer object.
		*/
		STPOpenGL::STPuint operator*() const;

		/**
		 * @brief Bind the current GL buffer to a target in the context.
		 * @param target The GL target being bound to.
		*/
		void bind(STPOpenGL::STPenum) const;

		/**
		 * @brief Bind a buffer object to an indexed buffer target.
		 * @param target Specify the target of the bind operation.
		 * @param Specify the index of the binding point within the array specified by target.
		*/
		void bindBase(STPOpenGL::STPenum, STPOpenGL::STPuint) const;

		/**
		 * @brief Map all of a buffer object's data store into the client's address space.
		 * @param access The flag for access policies.
		 * @return A pointer to the mapped buffer.
		*/
		void* mapBuffer(STPOpenGL::STPenum);

		/**
		 * @brief Map all or part of a buffer object's data store into the client's address space.
		 * @param offset Specifies the starting offset within the buffer of the range to be mapped.
		 * @param length Specifies the length of the range to be mapped.
		 * @param access Specifies a combination of access flags indicating the desired access to the mapped range.
		 * @return A pointer to the mapped buffer.
		*/
		void* mapBufferRange(STPOpenGL::STPintptr, size_t, STPOpenGL::STPbitfield);

		/**
		 * @brief Indicate modifications to a range of a mapped buffer.
		 * @param offset Specifies the start of the buffer subrange, in basic machine units.
		 * @param length Specifies the length of the buffer subrange, in basic machine units.
		*/
		void flushMappedBufferRange(STPOpenGL::STPintptr, size_t);

		/**
		 * @brief Release the mapping of a buffer object's data store into the client's address space.
		 * @return Generally it returns true.
		 * Unless the data store contents have become corrupt during the time the data store was mapped.
		*/
		STPOpenGL::STPboolean unmapBuffer() const;

		/**
		 * @brief Unbind GL buffer from the current context.
		 * @param target The GL target to be unbound.
		*/
		static void unbind(STPOpenGL::STPenum);

		/**
		 * @brief Allocate immutable storage with no data.
		 * @param size The number of byte to be allocated.
		 * @param flag Flags for immutable storage.
		*/
		void bufferStorage(size_t, STPOpenGL::STPbitfield);

		/**
		 * @brief Transfer data to the storage.
		 * @param data An array of data to be transferred.
		 * @param size The number of byte to be passed to the storage.
		 * @param offset Specifies the offset into the buffer object's data store where data replacement will begin, measured in bytes. 
		*/
		void bufferSubData(const void*, size_t, STPOpenGL::STPintptr);

		/**
		 * @brief Allocate immutable storage for a buffer and transfer data to the storage.
		 * @param data An array of data to be submitted.
		 * @param size The number of byte to be passed to the storage.
		 * @param flag Flags for immutable storage.
		*/
		void bufferStorageSubData(const void*, size_t, STPOpenGL::STPbitfield);

	};

}
#endif//_STP_BUFFER_H_