#pragma once
#ifndef _STP_CHUNK_STORAGE_H_
#define _STP_CHUNK_STORAGE_H_

//System ADT
#include  <memory>
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
	 * @brief STPChunkStorage stores all chunks in a sensible data structure
	*/
	class STPChunkStorage {
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
			size_t operator()(const vec2&) const;
		};
		//Hash table that stores the chunks by world position
		typedef std::unordered_map<vec2, std::unique_ptr<STPChunk>, STPHashvec2> STPChunkCache;

		//chunk storage
		//the key will be the x,z world position of each chunk
		STPChunkCache TerrainMap2D;

	public:

		/**
		 * @brief Init chunk storage
		*/
		STPChunkStorage();

		~STPChunkStorage();

		/**
		 * @brief Add a new chunk to the storage.
		 * @param chunkPos new chunk world position, make sure the pointer is dynamic
		 * @param chunk new chunk with data
		 * @return True if the chunk has been added, false if chunk exists thus it won't be inserted
		*/
		bool addChunk(vec2, STPChunk*);

		/**
		 * @brief Get the chunk by its world position
		 * @param chunkPos the chunk world position
		 * @return The chunk with specified world position, return null if not found
		*/
		STPChunk* getChunk(vec2);

		/**
		 * @brief Remove the chunk by its world position
		 * @param chunkPos the chunk world position
		 * @return True if the chunk with specified world position has removed, or false if not found. 
		 * The chunk will be effectively deleted and memory is freed, and no longer be available inside the chunk
		*/
		bool removeChunk(vec2);

		/**
		 * @brief Effectively clear the storage and free all used memory
		*/
		void clearChunk();

	};
}
#endif//_STP_CHUNK_STORAGE_H_

