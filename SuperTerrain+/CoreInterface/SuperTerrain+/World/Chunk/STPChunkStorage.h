#pragma once
#ifndef _STP_CHUNK_STORAGE_H_
#define _STP_CHUNK_STORAGE_H_

#include <SuperTerrain+/STPCoreDefine.h>
//System ADT
#include <unordered_map>
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
	class STP_API STPChunkStorage {
	public:

		//A pair indicate the status of in-place chunk addition
		typedef std::pair<bool, STPChunk*> STPChunkConstructed;

		/**
		 * @brief The hash function for the glm::vec2
		*/
		struct STPHashvec2 {
		private:

			std::hash<float> hasher;

			/**
			 * @brief Hash one value, combining with the previous hash value
			 * @param seed The previous seed value
			 * @param value The current value to be hashed
			*/
			void hashSeed(size_t&, float) const;

		public:

			/**
			 * @brief Hash function for the pair
			 * @param position The position represented by the vector
			 * @return The hash of the pair
			*/
			size_t operator()(const glm::vec2&) const;
		};

	private:

		//Hash table that stores the chunks by world position
		typedef std::unordered_map<glm::vec2, STPChunk, STPHashvec2> STPChunkCache;

		//chunk storage
		//the key will be the x,z world position of each chunk
		STPChunkCache TerrainMap2D;

	public:

		/**
		 * @brief Init chunk storage
		*/
		STPChunkStorage();

		STPChunkStorage(const STPChunkStorage&) = delete;

		STPChunkStorage(STPChunkStorage&&) noexcept = default;

		STPChunkStorage& operator=(const STPChunkStorage&) = delete;

		STPChunkStorage& operator=(STPChunkStorage&&) = default;

		~STPChunkStorage() = default;

		/**
		 * @brief Construct a new chunk in-place if not presented. Otherwise return the prsented chunk
		 * @param chunkPos The world position of the chunk
		 * @param mapSize The size of the map for the chunk
		 * @return If chunk is not presented, it's constructed with provided arguments and return true and the new pointer
		 * Otherwise, return false and the pointer to the original chunk
		*/
		STPChunkConstructed construct(glm::vec2, glm::uvec2);

		/**
		 * @brief Get the chunk by its world position
		 * @param chunkPos the chunk world position
		 * @return The chunk with specified position, return null if chunk not found
		*/
		STPChunk* operator[](glm::vec2);

		/**
		 * @brief Get the number of chunk in the storage
		 * @return The number of chunk in the storage
		*/
		size_t size() const;

		/**
		 * @brief Remove the chunk by its world position
		 * @param chunkPos the chunk world position
		 * @return True if the chunk with specified world position has removed, or false if not found. 
		 * The chunk will be effectively deleted and memory is freed, and no longer be available inside the chunk
		*/
		bool remove(glm::vec2);

		/**
		 * @brief Effectively clear the storage and free all used memory
		*/
		void clear();

	};
}
#endif//_STP_CHUNK_STORAGE_H_

