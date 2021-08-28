#pragma once
#ifndef _STP_MEMORY_POOL_H_
#define _STP_MEMORY_POOL_H_

#include <STPCoreDefine.h>
//System
#include <mutex>
#include <memory>
//Container
#include <list>
#include <queue>
#include <unordered_map>

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPMemoryPoolType denotes the type of memory pool to use
	*/
	enum class STPMemoryPoolType : unsigned char {
		//A regular pagable memory pool
		Regular = 0x00u,
		//A memory pool which allocates page-locked memory
		Pinned = 0x01u
	};

	/**
	 * @brief STPMemoryPool is a reusable memory pool for regular host and pinned memory.
	 * Pinned memory allocation is slower than pagable memory, it's the best to reuse it.
	 * It supports reusing memory with different size, but
	 * it's still best suited for reusing memory with the same size, or a limited range of memory size.
	 * A large range of memory size will shred system memory and may slow down the program.
	 * STPMemoryPool is also thread-safe.
	 * @tparam T The type of the memory pool
	*/
	template<STPMemoryPoolType T>
	class STP_API STPMemoryPool {
	private:

		/**
		 * @brief A memory deleter for different types.
		*/
		struct STPMemoryDeleter {
		public:

			void operator()(void*) const;

		};

		//Header contains information about the pointer
		typedef size_t STPHeader;
		//The size of the header for a returned memory unit, in byte
		constexpr static unsigned char HEADER_SIZE = static_cast<unsigned char>(sizeof(STPHeader));

		typedef std::unique_ptr<void, STPMemoryDeleter> STPMemoryUnit;
		//A memory pool contains memory blocks with the same size.
		typedef std::queue<STPMemoryUnit, std::list<STPMemoryUnit>> STPMemoryUnitPool;

		//All memory pool with different sizes
		std::unordered_map<size_t, STPMemoryUnitPool> Collection;

		std::mutex PoolLock;

		/**
		 * @brief Attach a header to the beginning of the memory block
		 * @param memory The pointer to the memory block, must be larger than the content size
		 * @param content The header of the content to be writtn
		*/
		static void encodeHeader(unsigned char*, STPHeader);

		/**
		 * @brief Detach a header from the beginning of the memory block.
		 * If there is no header presented, undefined behavioud.
		 * @param memory The pointer to the memory block, must contatins header in before the given pointer.
		 * @return content The header of the content.
		*/
		static typename STPHeader decodeHeader(unsigned char*);

	public:

		/**
		 * @brief Init a memory pool
		*/
		STPMemoryPool() = default;

		~STPMemoryPool() = default;

		STPMemoryPool(const STPMemoryPool&) = delete;

		STPMemoryPool(STPMemoryPool&&) = default;

		STPMemoryPool& operator=(const STPMemoryPool&) = delete;

		STPMemoryPool& operator=(STPMemoryPool&&) = default;

		/**
		 * @brief Request a memory unit with specified size.
		 * If memory with size has been allocated previously, no allocation will happen and it will return immediately.
		 * Otherwise new memory is allocated and will be bound to the current memory pool.
		 * @param size The size of the memory to be requested, in byte.
		 * @return The pointer to the memory
		*/
		void* request(size_t);
		
		/**
		 * @brief Release the previously-requested memory to the memory pool.
		 * It's an undefined behaviour if the memory is not allocated by the same memory pool.
		 * @param memory The memory to be returned
		*/
		void release(void*);

	};

	typedef STPMemoryPool<STPMemoryPoolType::Regular> STPRegularMemoryPool;
	typedef STPMemoryPool<STPMemoryPoolType::Pinned> STPPinnedMemoryPool;

}
#endif//_STP_MEMORY_POOL_H_