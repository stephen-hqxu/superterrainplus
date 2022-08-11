#pragma once
#ifndef _STP_WORLD_PIPELINE_H_
#define _STP_WORLD_PIPELINE_H_

#include <SuperTerrain+/STPCoreDefine.h>
#include <SuperTerrain+/STPOpenGL.h>

#include "../Environment/STPChunkSetting.h"
//Chunk
#include "./Chunk/STPChunk.h"
//World Generator
#include "./Diversity/STPBiomeFactory.h"
#include "./Chunk/STPHeightfieldGenerator.h"
#include "./Diversity/Texture/STPTextureFactory.h"
//Multithreading
#include "../Utility/STPThreadPool.h"

#include "../Utility/Memory/STPSmartDeviceObject.h"

//System
#include <memory>

//CUDA
#include <cuda_runtime.h>

namespace SuperTerrainPlus {

	/**
	 * @brief STPWorldPipeline is the master terrain generation pipeline that manages all generators and storage system, generate terrain and prepare data 
	 * for rendering. It loads up data from storage class, composing and preparing data for renderer. 
	 * If data is currently ready, map will be loaded to terrain map buffer; otherwise map generators are called for texture synthesis.
	*/
	class STP_API STPWorldPipeline {
	public:

		/**
		 * @brief STPPipelineSetup contains all information to initialise a world pipeline.
		*/
		struct STPPipelineSetup {
		public:

			//Biomemap Generator
			STPDiversity::STPBiomeFactory* BiomemapGenerator;
			//Heightfield Generator
			STPHeightfieldGenerator* HeightfieldGenerator;
			//Splatmap Generator
			STPDiversity::STPTextureFactory* SplatmapGenerator;

			const STPEnvironment::STPChunkSetting* ChunkSetting;
		};

		/**
		 * @brief STPWorldLoadStatus indicates a chunk loading status when the user is requesting a new chunk.
		*/
		enum class STPWorldLoadStatus : unsigned char {
			//The centre chunk location has no change, no operation is performed.
			Unchanged = 0x00u,
			//The centre chunk location has changed, and operation is already in progress.
			//The user can keep using the front buffer without being affected.
			BackBufferBusy = 0x01u,
			//Since last time the world is requested to reload, some operations have been finished.
			//Back and front buffer has been swapped with the new information loaded.
			Swapped = 0x02u,
			//There is already a pending chunk loading requesting and back buffer is still in used.
			//No more vacant buffer to perform asynchronous operation for another task.
			//This new request is then ignored.
			BufferExhaust = 0xFFu
		};

		/**
		 * @brief STPTerrainMapType specifies the type of terrain map to retrieve
		*/
		enum class STPTerrainMapType : unsigned char {
			Biomemap = 0x00u,
			Heightmap = 0x01u,
			Splatmap = 0x02u
		};

		const STPEnvironment::STPChunkSetting& ChunkSetting;

	private:

		/**
		 * @brief STPChunkLoaderStatus indicates the working status of the chunk loader.
		*/
		enum class STPChunkLoaderStatus : unsigned char {
			//No work is being processed.
			Free = 0x00u,
			//Some works are being done right now.
			Busy = 0x01u,
			//Was busy, but it has finished.
			Yield = 0x02u
		};

		//CUDA stream
		STPSmartDeviceObject::STPStream BufferStream;
		/**
		 * @brief STPGeneratorManager aims to send instructions to terrain generators when the pipeline requests for chunks and it is not ready in the storage.
		*/
		class STPGeneratorManager;
		std::unique_ptr<STPGeneratorManager> Generator;
		/**
		 * @brief STPMemoryManager manages memory usage for the world pipeline, such as heightmap data.
		*/
		class STPMemoryManager;
		std::unique_ptr<STPMemoryManager> Memory;

		//we do this in a little cheaty way, that if the chunk is loaded the first time this make sure the
		//currentCentralPos is different from this value the last world position of the central chunk of the entire
		//visible chunks
		glm::ivec2 LastCentreLocation;

		//async chunk loader
		STPThreadPool PipelineWorker;
		std::future<void> MapLoader;

		/**
		 * @brief Check if the map loader thread is busy.
		 * @return A chunk loader status value.
		*/
		STPChunkLoaderStatus isLoaderBusy();

	public:

		/**
		 * @brief Initialise world pipeline with pipeline stages loaded.
		 * @param setup The pointer to pipeline stages and settings.
		*/
		STPWorldPipeline(STPPipelineSetup&);

		STPWorldPipeline(const STPWorldPipeline&) = delete;

		STPWorldPipeline(STPWorldPipeline&&) = delete;

		STPWorldPipeline& operator=(const STPWorldPipeline&) = delete;

		STPWorldPipeline& operator=(STPWorldPipeline&&) = delete;

		~STPWorldPipeline();

		/**
		 * @brief Automatically load the chunks based on camera position, visible range and chunk size.
		 * It will handle whether the central chunk has been changed compare to last time it was called, 
		 * and determine whether or not to reload or continue waiting for loading for the current chunk.
		 * @param viewPos The world position of the viewer
		 * @return The load status.
		 * How the function should behave is indicated by the load status returned.
		*/
		STPWorldLoadStatus load(const glm::dvec3&);

		/**
		 * @brief Change the rendering chunk status to force reload terrain map onto the GL texture memory.
		 * If chunk position is not in the rendering range, command is ignored.
		 * It will insert current chunk into reloading queue and chunk will not be reloaded until the next rendering loop.
		 * When the rendering chunks are changed, all unprocessed queries are discarded as all new rendering chunks are reloaded regardless.
		 * @param chunkCoord The world coordinate of the chunk required for reloading
		 * @return True if query has been submitted successfully, false if chunk is not in the rendering range or the same query has been submitted before.
		*/
		bool reload(const glm::ivec2&);

		/**
		 * @brief Get the current world position of the central chunk.
		 * @return The pointer to the current central chunk.
		*/
		const glm::ivec2& centre() const noexcept;

		/**
		 * @brief Get the GL texture object containing the terrain map on the current front buffer.
		 * @type The type of terrain map to retrieve
		 * @return The GL object with the requesting terrain map.
		*/
		STPOpenGL::STPuint operator[](STPTerrainMapType) const noexcept;

		/**
		 * @brief Retrieve the pointer to splatmap generator.
		 * This can be used by the renderer for rule-based multi-biome terrain texture splatting.
		 * @return The splatmap generator.
		*/
		const STPDiversity::STPTextureFactory& splatmapGenerator() const noexcept;

	};

}
#endif//_STP_WORLD_PIPELINE_H_