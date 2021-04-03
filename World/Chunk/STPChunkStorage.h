#pragma once
#ifndef _STP_CHUNK_STORAGE_H_
#define _STP_CHUNK_STORAGE_H_

//System ADT
#include  <memory>
#include <unordered_map>
//Thread safety
#include <shared_mutex>
//Chunks
#include "STPChunk.h"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPChunkStorage stores all chunks in a sensible data structure.
	 * STPChunkStorage is thread safe.
	*/
	class STPChunkStorage {
	public:

		//A pair indicate the status of in-place chunk addition
		typedef std::pair<bool, STPChunk*> STPChunkConstructed;

	private:

		/**
		 * @brief The hash function for the glm::vec2
		*/
		struct STPHashvec2 {
		public:
			/**
			 * @brief Hash function for the pair
			 * @param position The position represented by the vector
			 * @return The hash of the pair
			*/
			size_t operator()(const glm::vec2&) const;
		};
		//Hash table that stores the chunks by world position
		typedef std::unordered_map<glm::vec2, std::unique_ptr<STPChunk>, STPHashvec2> STPChunkCache;

		//chunk storage
		//the key will be the x,z world position of each chunk
		STPChunkCache TerrainMap2D;
		
		//thread safety
		mutable std::shared_mutex chunk_storage_lock;

	public:

		/**
		 * @brief Init chunk storage
		*/
		STPChunkStorage();

		STPChunkStorage(const STPChunkStorage&) = delete;

		STPChunkStorage(STPChunkStorage&&) = delete;

		STPChunkStorage& operator=(const STPChunkStorage&) = delete;

		STPChunkStorage& operator=(STPChunkStorage&&) = delete;

		~STPChunkStorage();

		/**
		 * @brief Atomically add a new chunk to the storage.
		 * @param chunkPos new chunk world position, make sure the pointer is dynamic
		 * @param chunk new chunk with data
		 * @return True if the chunk has been added, false if chunk exists thus it won't be inserted
		*/
		bool addChunk(glm::vec2, STPChunk*);

		/**
		 * @brief Atomically construct a new chunk in-place if not presented. Otherwise return the prsented chunk
		 * @param chunkPos The world position of the chunk
		 * @param mapSize The size of the map for the chunk
		 * @return If chunk is not presented, it's constructed with provided arguments and return true and the new pointer
		 * Otherwise, return false and the pointer to the original chunk
		*/
		STPChunkConstructed constructChunk(glm::vec2, glm::uvec2);

		/**
		 * @brief Atomically get the chunk by its world position
		 * @param chunkPos the chunk world position
		 * @return The chunk with specified position, return null if chunk not found
		*/
		STPChunk* getChunk(glm::vec2);

		/**
		 * @brief Atomically remove the chunk by its world position
		 * @param chunkPos the chunk world position
		 * @return True if the chunk with specified world position has removed, or false if not found. 
		 * The chunk will be effectively deleted and memory is freed, and no longer be available inside the chunk
		*/
		bool removeChunk(glm::vec2);

		/**
		 * @brief Atomically and effectively clear the storage and free all used memory
		*/
		void clearChunk();

	};
}
#endif//_STP_CHUNK_STORAGE_H_

