#pragma once
#ifndef _STP_CHUNK_MANAGER_H_
#define _STP_CHUNK_MANAGER_H_

#include <SuperTerrain+/STPCoreDefine.h>
//GLM
#include <glm/gtc/type_ptr.hpp>
//OpenGL
#include <glad/glad.h>
//CUDA
#include <cuda_gl_interop.h>//used to upload to opengl texture in multithread
#include "../../Utility/STPSmartStream.h"

//Chunks
#include "STPChunkProvider.h"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPChunkManager acts as an interactive agent between chunk storage unit (STPChunk) and chunk renderer (STPProcedural2DINF). It loads up data from storage class,
	 * composing and preparing data for renderer. If data is currently ready, map will be loaded to opengl and send to renderer, if not ready, it will call CUDA functions and 
	 * generate maps.
	*/
	class STP_API STPChunkManager {
	public:

		//Vector that stored rendered chunk world position and loading status (True is loaded, false otherwise)
		typedef std::vector<std::pair<glm::vec2, bool>> STPLocalChunks;
		typedef std::unordered_map<glm::vec2, int, STPChunkStorage::STPHashvec2> STPLocalChunksTable;

		/**
		 * @brief STPRenderingBufferType specifies the type of rendering buffer to retrieve
		*/
		enum class STP_API STPRenderingBufferType : unsigned char {
			//Biomemap
			BIOME = 0x00u,
			//Heightmap and normalmap
			HEIGHTFIELD = 0x01u
		};

	private:

		//cuda stream
		STPSmartStream buffering_stream;

		//thread pool
		STPThreadPool compute_pool;

		//chunk data provider
		STPChunkProvider& ChunkProvider;

		//Heightfield
		//index 0: R16UI biome map
		//index 1: RGBA16 with normal map in RGB and heightmap in A
		GLuint terrain_heightfield[2];
		//registered buffer and texture
		cudaGraphicsResource_t heightfield_texture_res[2];
		//empty buffer (using cuda pinned memory) that is used to clear a chunk data, quad_clear is RGBA16
		unsigned short *quad_clear = nullptr;

		//async chunk loader
		std::future<unsigned int> ChunkLoader;

		//for automatic chunk loading
		//we do this in a little cheaty way, that if the chunk is loaded the first time this make sure the currentCentralPos is different from this value
		glm::vec2 lastCentralPos = glm::vec2(std::numeric_limits<float>::min());//the last world position of the central chunk of the entire visible chunks
		//Whenever camera changes location, all previous rendering buffers are dumpped
		bool trigger_clearBuffer;
		//determine which chunks to render and whether it's loaded, index of element denotes chunk local ID
		STPLocalChunks renderingLocals;
		STPLocalChunksTable renderingLocals_lookup;

		/**
		 * @brief Transfer rendering buffer on host side to device (OpenGL) rendering buffer by local chunk.
		 * @param buffer Rendering buffer on device side, a mapped OpenGL pointer.
		 * Rendering buffer is continuous, function will determine pointer offset and only chunk specified in the "image" argument will be updated.
		 * @param chunkPos World position of the chunk that will be used to update render buffer
		 * @param chunkID Local chunk ID that specified which chunk in rendering buffer will be overwritten.
		 * @return True if request has been submitted, false if given chunk is not available
		*/
		bool renderingBufferChunkSubData(cudaArray_t[2], glm::vec2, unsigned int);

		/**
		 * @brief Clear up the rendering buffer of the chunk map
		 * @param destination The loaction to store all loaded maps, and it will be erased.
		 * @param pixel_size The size of one pixel in byte
		*/
		void clearRenderingBuffer(cudaArray_t, size_t);

	public:

		/**
		 * @brief Init the chunk manager. Allocating spaces for opengl texture.
		 * @param provider The chunk provider link with this chunk manager
		*/
		STPChunkManager(STPChunkProvider&);

		~STPChunkManager();

		STPChunkManager(const STPChunkManager&) = delete;

		STPChunkManager(STPChunkManager&&) = delete;

		STPChunkManager& operator=(const STPChunkManager&) = delete;

		STPChunkManager& operator=(STPChunkManager&&) = delete;

		/**
		 * @brief Generate texture mipmap for 2 terrain texture
		*/
		void generateMipmaps();

		/**
		 * @brief Load the texture for chunks that are specified in the STPLocalChunks. The program will check if the maps are available for this chunk, and if so, texture will be loaded to the internal buffer
		 * and ready to transfer to chunk texture for rendering. Otherwise(chunk is new, there is no texture stored), compute will be dispatched and the chunkID for which
		 * chunks that are not available will remain in the STPLocalChunks, loaded chunks will be removed from the list. In order to achieve asynchronous loading,
		 * it is recommend to re-pass the same list everyframe so the program can check the status of computing chunks and load on demand.
		 * If the map is available and ready to use the engine will load the heightmap and normal map in the texture.
		 * Otherwise the it will start computing, and load next time the function get called.
		 * This is function is will start chunk loading asynchorously, so the main thread may return before any operation is finished.
		 * Before using the texture bounded to the heightmap, SyncloadChunks() needs to be called to make sure all used contexts will be loaded to prevent data racing.
		 * @param loading_chunks Specify the local chunkID and the position of each chunk in world coordinate, and the loader will only check and load the specified chunks to the
		 * texture array.
		 * If chunk is loaded on to the memory, it will be removed from the list. If compute is dispatched for that chunk, it will remain on the list. Passing an empty list
		 * will make the program do no preparation.
		 * Changing the original objects where the refernece is pointing to after loadChunksAsync() and before SyncloadChunks() is called will result in 
		 * undefined behaviour.
		 * If previous worker has yet finished, function will be blocked until the previous returned, then it will be proceed.
		 * @return True if loading worker has been dispatched, false if there is no chunks specified in the list.
		*/
		bool loadChunksAsync(STPLocalChunks&);

		/**
		 * @brief Similar to "void loadChunksAsync(STPLocalChunks&)", but will automatically load the chunks based on camera position, visible range and chunk size.
		 * It will handle whether the central chunk has been changed compare to last time it was called, and determine whether or not to reload or continue waiting for loading 
		 * for the current chunk.
		 * If previous worker has yet finished, function will be blocked until the previous returned, then it will be proceed.
		 * @param cameraPos The world position of the camera
		 * @return True if loading worker has been dispatched, false if there is no chunks specified in the list.
		*/
		bool loadChunksAsync(glm::vec3);

		/**
		 * @brief Change the rendering chunk status to force reload that will trigger a chunk texture reload onto rendering buffer
		 * Only when chunk position is being rendered, if chunk position is not in the rendering range, command is ignored.
		 * It will insert current chunk into reloading queue and chunk will not be reloaded until the next rendering loop.
		 * When the rendering chunks are changed, all un-processed queries are discarded as all new rendering chunks are reloaded regardlessly.
		 * @param chunkPos The world position of the chunk required for reloading
		 * @return True if query has been submitted successfully, false if chunk is not in the rendering range or the same query has been submitted before.
		*/
		bool reloadChunkAsync(glm::vec2);

		/**
		 * @brief Sync the loadChunksAsync() to make sure the work has finished before this function returns. 
		 * This function will be called automatically by the renderer before rendering.
		 * @return The number of chunk that has been reloaded in the last call to loadChunksAsync(). If all chunks have been loaded without issues, 0 will be returned.
		 * If no loadChunksAsync() worker has been dispatched, -1 will be returned.
		*/
		int SyncloadChunks();

		/**
		 * @brief Get the chunk provider
		 * @return The pointer to the chunk provider
		*/
		STPChunkProvider& getChunkProvider();

		/**
		 * @brief Get the current OpenGL rendering buffer.
		 * it's the rendering buffer for this frame.
		 * @type The type of rendering buffer to retrieve
		 * @return The current rendering buffer
		*/
		GLuint getCurrentRenderingBuffer(STPRenderingBufferType) const;

	};
}
#endif//_STP_CHUNK_MANAGER_H_