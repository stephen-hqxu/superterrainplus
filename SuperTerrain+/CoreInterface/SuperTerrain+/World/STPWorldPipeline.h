#ifndef _STP_WORLD_PIPELINE_H_
#define _STP_WORLD_PIPELINE_H_

#include <SuperTerrain+/STPCoreDefine.h>
#include <SuperTerrain+/STPOpenGL.h>
//Containers
#include <vector>
#include <list>
#include <queue>
#include <unordered_map>
//System
#include <memory>

#include "../Environment/STPChunkSetting.h"
//Chunk
#include "./Chunk/STPChunkStorage.h"
//World Generator
#include "./Diversity/STPBiomeFactory.h"
#include "./Chunk/STPHeightfieldGenerator.h"
#include "./Diversity/Texture/STPTextureFactory.h"
//Multithreading
#include "../Utility/STPThreadPool.h"
#include <shared_mutex>

//CUDA
#include <cuda_runtime.h>

namespace SuperTerrainPlus {

	/**
	 * @brief STPWorldPipeline is the master terrain generation pipeline that manages all generators and storage system, generate terrain and prepare data 
	 * for rendering. It loads up data from storage class, composing and preparing data for renderer. 
	 * If data is currently ready, map will be loaded to rendering buffer; otherwise map generators are called for texture synthesis.
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
			STPCompute::STPHeightfieldGenerator* HeightfieldGenerator;
			//Splatmap Generator
			STPDiversity::STPTextureFactory* SplatmapGenerator;

			const STPEnvironment::STPChunkSetting* ChunkSetting;
		};

		/**
		 * @brief STPRenderingBufferType specifies the type of rendering buffer to retrieve
		*/
		enum class STPRenderingBufferType : unsigned char {
			//Biomemap
			BIOME = 0x00u,
			//Heightmap
			HEIGHTFIELD = 0x01u,
			//Splatmap
			SPLAT = 0x02u
		};

	private:

		//The total number of rendering buffer we have currently.
		//Biomemap, heightfield and splatmap
		static constexpr size_t BufferCount = 3ull;

		//Vector that stored rendered chunk world position and loading status (True is loaded, false otherwise)
		typedef std::vector<std::pair<glm::vec2, bool>> STPLocalChunkStatus;
		//Use chunk world coordinate to lookup chunk ID
		typedef std::unordered_map<glm::vec2, unsigned int, STPChunkStorage::STPHashvec2> STPLocalChunkDictionary;

		/**
		 * @brief STPRenderingBufferMemory contains data to mapped GL texture data.
		*/
		struct STPRenderingBufferMemory {
		public:

			//Channel size in byte (not bit) for each map
			//Remember to update this value in case OpenGL buffer changes internal channel format
			constexpr static size_t Format[STPWorldPipeline::BufferCount] = {
				sizeof(STPDiversity::Sample),
				sizeof(unsigned short),
				sizeof(unsigned char)
			};

			//Map data
			cudaArray_t Map[STPWorldPipeline::BufferCount];

		};

		/**
		 * @brief STPRenderingBufferCache contains data as a backup of rendering buffer memory
		*/
		struct STPRenderingBufferCache {
		public:

			//Map cache data
			void* MapCache[STPWorldPipeline::BufferCount];
			//pitch size
			size_t Pitch[STPWorldPipeline::BufferCount];

			//A record of rendering locals for the current rendering buffer
			STPLocalChunkDictionary RenderingLocal;

		};

		STPThreadPool PipelineWorker;

		/**
		 * @brief STPGeneratorManager aims to send instructions to terrain generators when the pipeline requests for chunks and it is not ready in the storage.
		*/
		class STPGeneratorManager;
		std::unique_ptr<STPGeneratorManager> Generator;

		//cuda stream
		STPSmartStream BufferStream;

		//index 0: R16UI biome map
		//index 1: R16 height map
		//index 2: R8UI splat map
		STPOpenGL::STPuint TerrainMap[STPWorldPipeline::BufferCount];
		//registered buffer and texture
		cudaGraphicsResource_t TerrainMapRes[STPWorldPipeline::BufferCount];
		//empty buffer (using cuda pinned memory) that is used to clear a chunk data
		void* TerrainMapClearBuffer;
		//A cache that holds the previous rendered chunk memory to update the new rendered chunk
		STPRenderingBufferCache TerrainMapExchangeCache;

		//async chunk loader
		std::future<void> MapLoader;

		//for automatic chunk loading
		//we do this in a little cheaty way, that if the chunk is loaded the first time this make sure the currentCentralPos is different from this value
		glm::vec2 lastCenterLocation = glm::vec2(std::numeric_limits<float>::min());//the last world position of the central chunk of the entire visible chunks
		//Whenever camera changes location, all previous rendering buffers are dumpped
		bool shouldClearBuffer;
		//determine which chunks to render and whether it's loaded, index of element denotes chunk local ID
		STPLocalChunkStatus renderingLocal;
		STPLocalChunkDictionary renderingLocalLookup;

		/**
		 * @brief Copy from a portion of rendering buffer to another portion of rendering buffer.
		 * @param dest The destination of the rendering buffer to be copied to.
		 * @param dest_idx The local chunk index to the destination rendering sub-buffer.
		 * @param src_idx The local chunk index to the source rendering sub-buffer.
		*/
		void copySubBufferFromSubCache(const STPRenderingBufferMemory&, unsigned int, unsigned int);

		/**
		 * @brief Copy the given rendering buffer to a backup buffer in the current pipeline.
		 * @param buffer The rendering buffer to be backed-up.
		*/
		void backupBuffer(const STPRenderingBufferMemory&);

		/**
		 * @brief Clear up the rendering buffer of the chunk map
		 * @param destination The loaction to store all loaded maps, and it will be erased.
		*/
		void clearBuffer(const STPRenderingBufferMemory&);

		/**
		 * @brief Transfer rendering buffer on host side to device (OpenGL) rendering buffer by local chunk.
		 * @param buffer Rendering buffer on device side, a mapped OpenGL pointer.
		 * Rendering buffer is continuous, function will determine pointer offset and only chunk specified in the "image" argument will be updated.
		 * @param chunkPos World position of the chunk that will be used to update render buffer
		 * @param chunkID Local chunk ID that specified which chunk in rendering buffer will be overwritten.
		 * @return True if request has been submitted, false if given chunk is not available.
		*/
		bool mapSubData(const STPRenderingBufferMemory&, glm::vec2, unsigned int);

		/**
		 * @brief Get the chunk offset on a rendering buffer given a local chunk index.
		 * @param index The local chunk index.
		 * @return The chunk index offset on the rendering buffer.
		*/
		glm::uvec2 calcBufferOffset(unsigned int) const;

	public:

		const STPEnvironment::STPChunkSetting& ChunkSetting;

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
		 * It will handle whether the central chunk has been changed compare to last time it was called, and determine whether or not to reload or continue waiting for loading
		 * for the current chunk.
		 * If previous worker has yet finished, function will be blocked until the previous returned, then it will be proceed.
		 * Loading must be sync with wait() or it will incur undefined behaviour.
		 * @param cameraPos The world position of the camera
		 * @return True if loading worker has been dispatched, false if there is no chunk need to be updated.
		*/
		bool load(glm::vec3);

		/**
		 * @brief Change the rendering chunk status to force reload that will trigger a chunk texture reload onto rendering buffer.
		 * Only when chunk position is being rendered, if chunk position is not in the rendering range, command is ignored.
		 * It will insert current chunk into reloading queue and chunk will not be reloaded until the next rendering loop.
		 * When the rendering chunks are changed, all un-processed queries are discarded as all new rendering chunks are reloaded regardlessly.
		 * @param chunkPos The world position of the chunk required for reloading
		 * @return True if query has been submitted successfully, false if chunk is not in the rendering range or the same query has been submitted before.
		*/
		bool reload(glm::vec2);

		/**
		 * @brief Sync the map loading operations to make sure the work has finished before this function returns.
		 * This function should be called by the renderer before rendering to ensure data safety.
		*/
		void wait();

		/**
		 * @brief Get the current OpenGL rendering buffer.
		 * it's the rendering buffer for this frame.
		 * @type The type of rendering buffer to retrieve
		 * @return The current rendering buffer
		*/
		STPOpenGL::STPuint operator[](STPRenderingBufferType) const;

		/**
		 * @brief Retrieve the pointer to splatmap generator.
		 * This can be used by the renderer for rule-based multi-biome terrain texture splatting.
		 * @return The splatmap generator.
		*/
		STPDiversity::STPTextureFactory& splatmapGenerator() const;

	};

}
#endif//_STP_WORLD_PIPELINE_H_