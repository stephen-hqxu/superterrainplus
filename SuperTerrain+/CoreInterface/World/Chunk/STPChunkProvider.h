#pragma once
#ifndef _STP_CHUNK_PROVIDER_H_
#define _STP_CHUNK_PROVIDER_H_

#include <STPCoreDefine.h>
//System
#include <utility>
#include <functional>

//Multithreading
#include "../../Utility/STPThreadPool.h"
//Chunks
#include "STPChunkStorage.h"
//2D terrain compute engine
#include "../Diversity/STPBiomeFactory.h"
#include "../../GPGPU/STPHeightfieldGenerator.cuh"

//Settings
#include "../../Environment/STPConfiguration.h"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPChunkProvider read chunks from chunk storage, and if the chunk is not available, it will return some status and dispatch compute async accordingly
	*/
	class STP_API STPChunkProvider {
	private:

		//Chunk settings
		const STPEnvironment::STPChunkSetting& ChunkSetting;
		//chunk data
		STPChunkStorage& ChunkStorage;
		//thread pool
		std::unique_ptr<STPThreadPool> kernel_launch_pool;
		//Biomemap generator
		STPDiversity::STPBiomeFactory& generateBiome;
		//Heightfield generator
		STPCompute::STPHeightfieldGenerator& generateHeightfield;

		typedef std::list<STPChunk*> STPChunkNeighbour;

		/**
		 * @brief Calculate the chunk offset such that the transition of each chunk is seamless
		 * @param chunkPos The world position of the chunk
		 * @return The chunk offset
		*/
		float2 calcChunkOffset(glm::vec2) const;

		/**
		 * @brief Get all neighbour for this chunk position
		 * @param chunkPos The central chunk world position
		 * @return A list of all neighbour chunk position
		*/
		STPChunk::STPChunkPositionCache getNeighbour(glm::vec2) const;

		/**
		 * @brief Dispatch compute for heightmap, the heightmap result will be writen back to the storage
		 * @param current_chunk The maps for the chunk
		 * @param neighbour_chunks The maps of the chunks that require to be used for biome-edge interpolation during heightmap generation, 
		 * require the central chunk and neighbour chunks arranged in row-major flavour. The central chunk should also be included.
		 * @param chunkPos The world position of the chunk
		*/
		void computeHeightmap(STPChunk*, STPChunkProvider::STPChunkNeighbour&, glm::vec2);

		/**
		 * @brief Dispatch compute for free-slip hydraulic erosion, normalmap compute and formatting, requires heightmap presenting in the chunk
		 * @param neighbour_chunks The maps of the chunks that require to be eroded with a free-slip manner, require the central chunk and neighbour chunks 
		 * arranged in row-major flavour. The central chunk should also be included
		*/
		void computeErosion(STPChunkProvider::STPChunkNeighbour&);

		/**
		 * @brief Recursively prepare neighbour chunks for the central chunk.
		 * The first recursion will prepare neighbour biomemaps for heightmap generation.
		 * The second recursion will prepare neighbour heightmaps for erosion.
		 * @param chunkPos The position to the chunk which should be prepared.
		 * @param erosion_reloader The function to call when erosion has been performed so rendering buffer for neighbour chunks will be checked later
		 * @param rec_depth Please leave this empty, this is the recursion depth and will be managed properly
		 * @return If all neighbours are ready to be used, true is returned.
		 * If any neighbour is not ready (being used by other threads or neighbour is not ready and compute is launched), return false
		*/
		bool prepareNeighbour(glm::vec2, std::function<bool(glm::vec2)>&, unsigned char = 2u);

	public:

		/**
		 * @brief Calculate the maximum number of chunk that can be computed in parallel without triggering chunk overlap and data race
		 * @param rendered_range The number of chunk to be rendered
		 * @param freeslip_range The numebr of chunk that will be used as neighbour
		*/
		static unsigned int calculateMaxConcurrency(glm::uvec2, glm::uvec2);

		/**
		 * @brief Init the chunk provider.
		 * @param chunk_settings All settings about chunk to be linked with this provider
		 * @param storage The storage unit to link with
		 * @param biome_factory The biomemap factory/generator to link with
		 * @param heightfield_generator The heightfield generator to link with
		*/
		STPChunkProvider(const STPEnvironment::STPChunkSetting&, STPChunkStorage&, STPDiversity::STPBiomeFactory&, STPCompute::STPHeightfieldGenerator&);

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
		const STPEnvironment::STPChunkSetting& getChunkSetting() const;

	};
}
#endif//_STP_CHUNK_PROVIDER_H_
