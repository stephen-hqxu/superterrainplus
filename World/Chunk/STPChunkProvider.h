#pragma once
#ifndef _STP_CHUNK_PROVIDER_H_
#define _STP_CHUNK_PROVIDER_H_

//System ADT
#include <utility>
//Multi-threading
#include "../../Helpers/STPThreadPool.h"

//Chunks
#include "STPChunkStorage.h"
//2D terrain compute engine
#include "../../GPGPU/STPHeightfieldGenerator.cuh"
#include "../../GPGPU/STPImageConverter.cuh"

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
	public:

		/**
		 * @brief STPChunkReadyStatus indicates the ready status when the chunk is requested
		*/
		enum class STPChunkReadyStatus: unsigned char {
			//Chunk cannot be found in the provided storage unit, computation has been dispatched async, result will be automatically written back to the storage
			Not_Found = 0x00,
			//The nominated chunk is currently locked by the other thread, nothing will be done and return instantly
			In_Used = 0x01,
			//Chunk is ready and it will be returned now
			Complete = 0x02
		};

		//The return value of the chunk request, if the status indicates that the chunk is not available, null will be returned for the chunk field
		//Otherwise a pointer to the chunk, with respect to the provided chunk storage unit, will be returned
		typedef std::pair<STPChunkReadyStatus, STPChunk*> STPChunkLoaded;

	private:

		//thread safety
		mutable std::shared_mutex chunk_storage_locker;
		//thread pool
		STPThreadPool* const compute_pool;

		//Chunk settings
		const STPSettings::STPChunkSettings ChunkSettings;

		//Heightfield generator
		STPCompute::STPHeightfieldGenerator* heightmap_gen = nullptr;
		//formatter
		STPCompute::STPImageConverter* formatter = nullptr;

		/**
		 * @brief Dispatch compute for 2d terrain asynchornously, the results will be written back to chunk storage
		 * @param current_chunk The maps for the chunk that needs to be loaded and computed
		 * @param chunkPos The world position of this chunk, this acts as a key in the look up table
		 * @return True if all maps are computed and returned back to data storage.
		*/
		bool computeChunk(STPChunk* const, glm::vec2);

		/**
		 * @brief Format the terrain image format, currently it will convert 32bit per channel to 16bit. 
		 * Such that we can keep an original copy of the maps for further calculation.
		 * Format will not happen if the current chunk is being used by other threads or it has been formatted already
		 * @param current_chunk The maps for the chunk that needs to be converted
		 * @return True if all maps are computed, and chunk lock will be released with updated chunk status. 
		 * False if chunk is not completed or the memory is up-to-date.
		*/
		bool formatChunk(STPChunk* const);

	public:

		/**
		 * @brief Init the chunk provider
		 * @param settings Stores all settings for terrain generation
		 * @param shared_threadpool The thread pool for multithreading computing and rendering. It can share the pool with the main program or separate from.
		*/
		STPChunkProvider(STPSettings::STPConfigurations*, STPThreadPool* const);

		~STPChunkProvider();

		STPChunkProvider(const STPChunkProvider&) = delete;

		STPChunkProvider(STPChunkProvider&&) = delete;

		STPChunkProvider& operator=(const STPChunkProvider&) = delete;

		STPChunkProvider& operator=(STPChunkProvider&&) = delete;

		/**
		 * @brief Request the texture maps in the given chunk storage if they can be found on library, otherwise compute will be dispatched
		 * @param source The location where to load chunks from
		 * @param chunkPos The world position of this chunk, this acts as a key in the look up table
		 * @return A pair of chunk ready status and the pointer to the chunk (respect to the provided chunk storage)
		*/
		STPChunkLoaded requestChunk(STPChunkStorage*, glm::vec2);

		/**
		 * @brief Get the chunk settings
		 * @return The chunk settings
		*/
		const STPSettings::STPChunkSettings* getChunkSettings() const;
		
		/**
		 * @brief Set the number of iteration each heightfield generation will use
		 * @param iteration the number of iteration
		 * @return True if set
		*/
		bool setHeightfieldErosionIteration(unsigned int);

	};
}
#endif//_STP_CHUNK_PROVIDER_H_
