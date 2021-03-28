#pragma once
#ifndef _STP_MEMORY_POOL_HPP_
#define _STP_MEMORY_POOL_HPP_

//ADT
#include <queue>
#include <unordered_map>
#include <type_traits>

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPClassicMemoryManager is the default allocator and deallocator
	 * @tparam T The type of the object to allocate memory
	*/
	template<typename T>
	class STPClassicMemoryManager {
	public:

		T* allocate(size_t size) {
			return new T[size];
		}

		void deallocate(size_t size, T* ptr) {
			delete[] ptr;
		}

	};

	/**
	 * @brief STPMemoryPool is a pool that stores allocated memories, sending out if request, storing it if no longer required instead of reallocating.
	 * Whenever a new memory is requested, it will first check if any more free memory available, or allocate new blocks of memory.
	 * STPMemoryPool is designed for non-contiguous memory blocks.
	 * @tparam T The type of object to allocate memory
	 * @tparam A The allocator for memory pool. See the allocate() and deallocate() function to learn the type signature
	*/
	template<typename T, class A = STPClassicMemoryManager<T>>
	class STPMemoryPool {
	private:

		//memory that is allcated but not in-use
		size_t availiable = 0ull;
		//The memory pool, similar to a multimap, but I need to keep FIFO rule
		std::unordered_map<size_t, std::queue<T*>> memory;
		//The allocator to create new memory block for that type
		A allocator;

		/**
		 * @brief Free memory by iterator, the memory queue that the iterator is pointing to is freed.
		 * The bucket that stores pointers with the same size will also be freed
		 * @tparam Ite Iterator of the container
		 * @param it Iterator. When the current container is freed, iterator will be advanced by itself
		*/
		template<typename Ite>
		void free(Ite& it) {
			if (it != this->memory.end()) {
				std::queue<T*>& block = it->second;
				while (!block.empty()) {
					//clear up the memory with this size
					this->free(it->first, block.front());
					block.pop();
					this->availiable--;
				}

				it = this->memory.erase(it);
			}
		}

	public:

		/**
		 * @brief Init STPMemoryPool
		*/
		STPMemoryPool() = default;

		~STPMemoryPool() {
			this->free();
		}

		/**
		 * @brief Return the number of memory elements that are available in the pool in total
		 * @return The number of allocated elements
		*/
		inline size_t size() const {
			return this->availiable;
		}

		/**
		 * @brief Return the number of memory with the specified size
		 * @param size The size of the memory block to find
		 * @return The number of memory with this size
		*/
		inline size_t size(size_t size) const {
			//using iterator to avoid creating a new queue object by unordered map
			auto it = this->memory.find(size);
			if (it != this->memory.end()) {
				return it->second.size();
			}
			return 0ull;
		}

		/**
		 * @brief Check if there is any allocated elements available
		 * @return True if there is no available memory
		*/
		inline bool empty() const {
			return this->size() == 0ull;
		}

		/**
		 * @brief Check if there is any memory with the size specified that is available right now
		 * @param size The size of the memory block
		 * @return True if there is no available memory for this size
		*/
		inline bool empty(size_t size) const {
			return this->size(size) == 0ull;
		}

		/**
		 * @brief Allocate new memory for the element.
		 * If there are any free memory stayed in the memory pool, it will be popped from the memory pool.
		 * Otherwise new memory will be reallocated.
		 * Memory must be freed using deallocate()
		 * @tparam ...Arg Arguments for user defined manipulations to the new memory
		 * @prarm size The amount of memory for this type to allocate
		 * @param ...arg Arguments for user defined manipulations to the new memory
		 * @return The newly allocated memory location
		*/
		template<typename... Arg>
		T* allocate(size_t size, Arg&&... arg) {
			//find available memory with this size
			std::queue<T*>& block = this->memory[size];
			if (block.empty()) {
				//if there is no available memory, allocate some new memory
				return this->allocator.allocate(size, std::forward<Arg>(arg)...);
			}

			//otherwise, pop one available memory
			T* ptr = block.front();
			block.pop();
			this->availiable--;
			return ptr;
		}

		/**
		 * @brief Preallocate memory with defined memory block size and count
		 * @tparam ...Arg Arguments for user defined manipulations to the new memory
		 * @param size The size of the memory block to be allocated
		 * @param count The number of allocation, each with the same size as defined
		 * @param ...arg Arg Arguments for user defined manipulations to the new memory
		*/
		template<typename... Arg>
		void preallocate(size_t size, size_t count, Arg&&... arg) {
			std::queue<T*>& block = this->memory[size];

			for (size_t i = 0ull; i < count; i++) {
				block.push(this->allocator.allocate(size, std::forward<Arg>(arg)...));
			}
		}

		/**
		 * @brief Deallocate the pointer allocated by allocate()
		 * Internally, it won't acutally be freed up, but stored and pop-out once allocate() is called next time
		 * @prarm size Amount of memory to return, it must have the same size as when it's called in allocate() function
		 * @param ptr The pointer to the memory that is allocated by allocate()
		*/
		void deallocate(size_t size, T* ptr) {
			//simply push back to our memory pool
			//if entry with this size not exist, if will create one itself
			this->memory[size].push(ptr);
			this->availiable++;
		}

		/**
		 * @brief Requests the container to reduce its capacity to fit its size.
		 * It takes O(n) where n is the number of memory with different size
		*/
		void shrink_to_fit() {
			for (auto it = this->memory.begin(); it != this->memory.end();) {
				std::queue<T*>& block = it->second;
				if (block.empty()) {
					it = this->memory.erase(it);
				}
				else {
					it++;
				}
			}
		}

		/**
		 * @brief Free one pointer that's allocated by allocate().
		 * @prarm size Amount of memory to return, it must have the same size as when it's called in allocate() function
		 * @param ptr The pointer to be freed
		*/
		void free(size_t size, T* ptr) {
			this->allocator.deallocate(size, ptr);
		}

		/**
		 * @brief Free up all memory in the memory pool with the specified size.
		 * If memory with size not found, nothing will be done.
		 * If memory with size has been found, but with no underlying pointers in the container, container will be freed
		 * @param size The size of the memory to be freed
		*/
		void free(size_t size) {
			auto it = this->memory.find(size);
			this->free(it);
		}

		/**
		 * @brief Free up all allocated memories in the memory pool
		*/
		void free() {
			for (auto it = this->memory.begin(); it != this->memory.end(); this->free(it)) {
				//we need to refresh the pointer to the beginning since the free(size) function will delete the entry
			}
		}

	};

}
#endif//_STP_MEMORY_POOL_HPP_