#pragma once
#ifndef _STP_WORLD_PIPELINE_H_
#define _STP_WORLD_PIPELINE_H_

#include <SuperTerrain+/STPCoreDefine.h>
#include <SuperTerrain+/STPOpenGL.h>

#include "../Environment/STPChunkSetting.h"
//World Generator
#include "./Diversity/STPBiomeFactory.h"
#include "./Chunk/STPHeightfieldGenerator.h"
#include "./Diversity/Texture/STPTextureFactory.h"
//Multithreading
#include "../Utility/STPThreadPool.h"

#include "../Utility/Memory/STPSmartDeviceObject.h"

//System
#include <memory>

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

			//The chunk setting will be copied under the world pipeline.
			const STPEnvironment::STPChunkSetting* ChunkSetting;
		};

		/**
		 * @brief STPWorldLoadStatus indicates a chunk loading status when the user is requesting a new chunk.
		*/
		enum class STPWorldLoadStatus : unsigned char {
			//The centre chunk location has no change, no operation is performed.
			Unchanged = 0x00u,
			//The back buffer is current busy on generation operations.
			//The user can keep using the front buffer without being affected, this chunk loading request is ignored.
			BackBufferBusy = 0x01u,
			//Since last time the world is requested to reload, some operations have been finished.
			//Back and front buffer has been swapped with the new information loaded.
			Swapped = 0x02u
		};

		/**
		 * @brief STPTerrainMapType specifies the type of terrain map to retrieve
		*/
		enum class STPTerrainMapType : unsigned char {
			Biomemap = 0x00u,
			Heightmap = 0x01u,
			Splatmap = 0x02u
		};

		const STPEnvironment::STPChunkSetting ChunkSetting;

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

		//The centre chunk world coordinate last time the map loader worked on.
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
		STPWorldPipeline(const STPPipelineSetup&);

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