#pragma once
#ifndef _STP_CHUNK_PROVIDER_H_
#define _STP_CHUNK_PROVIDER_H_

//System
#include <utility>
#include <functional>

//Multithreading
#include "../../Helpers/STPThreadPool.h"
//Chunks
#include "STPChunkStorage.h"
//2D terrain compute engine
#include "../../GPGPU/STPHeightfieldGenerator.cuh"

//Settings
#include "../../Settings/STPConfigurations.hpp"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPChunkProvider read chunks from chunk storage, and if the chunk is not available, it will return some status and dispatch compute async accordingly
	*/
	class STPChunkProvider {
	private:

		//Chunk settings
		const STPSettings::STPChunkSettings ChunkSettings;
		//chunk data
		STPChunkStorage ChunkCache;
		//thread pool
		std::unique_ptr<STPThreadPool> kernel_launch_pool;
		//Heightfield generator
		STPCompute::STPHeightfieldGenerator heightmap_gen;
		const unsigned int concurrency_level;

		/**
		 * @brief Calculate the maximum number of chunk that can be computed in parallel without triggering chunk overlap and data race
		 * @param rendered_range The number of chunk to be rendered
		 * @param freeslip_rance The numebr of chunk that will be used as neighbour
		*/
		static unsigned int calculateMaxConcurrency(glm::uvec2, glm::uvec2);

		/**
		 * @brief Calculate the chunk offset such that the transition of each chunk is seamless
		 * @param chunkPos The world position of the chunk
		 * @return The chunk offset
		*/
		float3 calcChunkOffset(glm::vec2) const;

		/**
		 * @brief Dispatch compute for heightmap, the heightmap result will be writen back to the storage
		 * @param current_chunk The maps for the chunk
		 * @param chunkPos The world position of the chunk
		*/
		void computeHeightmap(STPChunk*, glm::vec2);

		/**
		 * @brief Dispatch compute for free-slip hydraulic erosion, normalmap compute and formatting, requires heightmap presenting in the chunk
		 * @param current_chunk The central chunk for the computation
		 * @param neighbour_chunks The maps of the chunks that require to be eroded with a free-slip manner, require the central chunk and neighbour chunks 
		 * arranged in row-major flavour. The central chunk should also be included
		*/
		void computeErosion(STPChunk*, std::list<STPChunk*>&);

	public:

		/**
		 * @brief Init the chunk provider
		 * @param settings Stores all settings for terrain generation
		*/
		STPChunkProvider(STPSettings::STPConfigurations*);

		~STPChunkProvider() = default;

		/**
		 * @brief For every neighbour of the current chunk, check if they exists in the storage.
		 * If not found, heightmap compute will be launched.
		 * Then, all neighbours will be checked again to see if any is in used.
		 * If all neighbours are availble, and the center chunk is incomplete, lock all chunks and perform free-slip hydraulic erosion.
		 * Otherwise do nothing and return true.
		 * After this function call, the request chunk will be guaranteed to be completed and ready for rendering.
		 * @param chunkPos The world position of the center chunk
		 * @param reload_callback Used to trigget a chunk rendering reload in STPChunkManager
		 * @return True if the center chunk is complete, false if center chunk is incomplete and any of the neighbour chunks are in used.
		*/
		bool checkChunk(glm::vec2, std::function<bool(glm::vec2)>);

		/**
		 * @brief Request the texture maps in the given chunk storage if they can be found on library
		 * @param chunkPos The world position of this chunk, this acts as a key in the look up table
		 * @return The pointer to the chunk if it's available and not in used, otherwise nullptr
		*/
		STPChunk* requestChunk(glm::vec2);

		/**
		 * @brief Get the chunk settings
		 * @return The chunk settings
		*/
		const STPSettings::STPChunkSettings* getChunkSettings() const;

	};
}
#endif//_STP_CHUNK_PROVIDER_H_
