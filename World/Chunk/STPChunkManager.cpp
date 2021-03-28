#pragma warning(disable : 4267)//implicit conversion may lose data wwarning (actually it won't)
#include "STPChunkManager.h"

using glm::vec2;
using glm::vec4;
using glm::value_ptr;

using namespace SuperTerrainPlus;

STPChunkManager::STPChunkManager(STPSettings::STPConfigurations* settings, STPThreadPool* const shared_threadpool) : compute_pool(shared_threadpool)
	, ChunkCache(), ChunkProvider(settings, shared_threadpool) {

	const STPSettings::STPChunkSettings* chunk_settings = this->ChunkProvider.getChunkSettings();
	const int chunk_num = static_cast<int>(chunk_settings->RenderedChunk.x * chunk_settings->RenderedChunk.y);
	const int totaltexture_size = chunk_num * static_cast<int>(chunk_settings->MapSize.x * chunk_settings->MapSize.y) * sizeof(unsigned short);//one channel

	//creating texture
	glCreateTextures(GL_TEXTURE_2D_ARRAY, 2, this->terrain_heightfield);
	//and allocating spaces for heightmap
	glTextureParameteri(*(this->terrain_heightfield), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(*(this->terrain_heightfield), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTextureParameteri(*(this->terrain_heightfield), GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTextureParameteri(*(this->terrain_heightfield), GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTextureParameteri(*(this->terrain_heightfield), GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTextureStorage3D(*(this->terrain_heightfield), 1, GL_R16, static_cast<int>(chunk_settings->MapSize.x), static_cast<int>(chunk_settings->MapSize.y), chunk_num);
	cudaGraphicsGLRegisterImage(&(this->heightfield_texture_res[0]), *(this->terrain_heightfield), GL_TEXTURE_2D_ARRAY, cudaGraphicsRegisterFlagsNone);
	//normal map is a bit unusual in term of color format
	glTextureParameteri(*(this->terrain_heightfield + 1), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(*(this->terrain_heightfield + 1), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTextureParameteri(*(this->terrain_heightfield + 1), GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTextureParameteri(*(this->terrain_heightfield + 1), GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTextureParameteri(*(this->terrain_heightfield + 1), GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTextureStorage3D(*(this->terrain_heightfield + 1), 1, GL_RGBA16, static_cast<int>(chunk_settings->MapSize.x), static_cast<int>(chunk_settings->MapSize.y), chunk_num);
	cudaGraphicsGLRegisterImage(&(this->heightfield_texture_res[1]), *(this->terrain_heightfield + 1), GL_TEXTURE_2D_ARRAY, cudaGraphicsRegisterFlagsNone);

	//init clear buffers that are used to clear texture when new rendered chunks are loaded (we need to clear the previous chunk data)
	cudaMallocHost(&this->mono_clear, totaltexture_size);
	cudaMallocHost(&this->quad_clear, totaltexture_size * 4);
	memset(this->mono_clear, 0xFF, totaltexture_size);
	memset(this->quad_clear, 0xFF, totaltexture_size * 4);
}

STPChunkManager::~STPChunkManager() {
	//wait until the worker has finished
	if (this->ChunkLoader.valid()) {
		this->ChunkLoader.get();
	}

	//unregister resource
	cudaGraphicsUnregisterResource(this->heightfield_texture_res[0]);
	cudaGraphicsUnregisterResource(this->heightfield_texture_res[1]);
	//delete texture
	glDeleteTextures(2, this->terrain_heightfield);

	//delete clear buffer
	cudaFreeHost(this->mono_clear);
	cudaFreeHost(this->quad_clear);
}

bool STPChunkManager::MapLoader(vec2 chunkPos, const cudaArray_t destination[2]) {
	const STPSettings::STPChunkSettings* chunk_settings = this->ChunkProvider.getChunkSettings();

	//ask provider if we can get the chunk
	auto result = this->ChunkProvider.requestChunk(this->ChunkCache, chunkPos);

	if (result.first == STPChunkProvider::STPChunkReadyStatus::Complete) {
		//chunk is ready, we can start loading
		STPChunk* const chunk = result.second;
		cudaStream_t copy_stream;
		//streaming copy
		bool no_error = true;
		no_error &= cudaSuccess == cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
		//host array does not have pitch so pitch = width
		no_error &= cudaSuccess == cudaMemcpy2DToArrayAsync(destination[0], 0, 0, chunk->getCacheMap(STPChunk::STPMapType::Heightmap),
			static_cast<int>(chunk->getSize().x) * sizeof(unsigned short), static_cast<int>(chunk->getSize().x) * sizeof(unsigned short),
			static_cast<int>(chunk->getSize().y), cudaMemcpyHostToDevice, copy_stream);
		no_error &= cudaSuccess == cudaMemcpy2DToArrayAsync(destination[1], 0, 0, chunk->getCacheMap(STPChunk::STPMapType::Normalmap),
			static_cast<int>(chunk->getSize().x) * sizeof(unsigned short) * 4, static_cast<int>(chunk->getSize().x) * sizeof(unsigned short) * 4,
			static_cast<int>(chunk->getSize().y), cudaMemcpyHostToDevice, copy_stream);
		no_error &= cudaSuccess == cudaStreamSynchronize(copy_stream);
		no_error &= cudaSuccess == cudaStreamDestroy(copy_stream);

		return no_error;
	}

	//not ready yet
	//clear unloaded chunk, so the engine won't display the chunk from previous rendered chunks
	cudaStream_t clear_stream;
	cudaStreamCreateWithFlags(&clear_stream, cudaStreamNonBlocking);
	cudaMemcpy2DToArrayAsync(destination[0], 0, 0, this->mono_clear,
		static_cast<int>(chunk_settings->MapSize.x) * sizeof(unsigned short), static_cast<int>(chunk_settings->MapSize.x) * sizeof(unsigned short),
		static_cast<int>(chunk_settings->MapSize.y), cudaMemcpyHostToDevice, clear_stream);
	cudaMemcpy2DToArrayAsync(destination[1], 0, 0, this->quad_clear,
		static_cast<int>(chunk_settings->MapSize.x) * sizeof(unsigned short) * 4, static_cast<int>(chunk_settings->MapSize.x) * sizeof(unsigned short) * 4,
		static_cast<int>(chunk_settings->MapSize.y), cudaMemcpyHostToDevice, clear_stream);
	cudaStreamSynchronize(clear_stream);
	cudaStreamDestroy(clear_stream);

	return false;
}

void STPChunkManager::generateMipmaps() {
	glGenerateTextureMipmap(*(this->terrain_heightfield));
	glGenerateTextureMipmap(*(this->terrain_heightfield + 1));
	return;
}

bool STPChunkManager::loadChunksAsync(STPLocalChunks& loading_chunks) {
	//waiting for the previous worker to finish(if any)
	//make sure there isn't any worker accessing the loading_chunks, otherwise undefined behaviour warning
	this->SyncloadChunks();
	if (loading_chunks.size() == 0) {
		return false;
	}

	//async chunk loader
	auto asyncChunkLoader = [this, &loading_chunks](std::list<std::unique_ptr<cudaArray_t[]>>& heightfield) -> int {
		static std::queue<std::future<bool>, std::list<std::future<bool>>> loader;
		//A functoin to load one chunk
		//return true if chunk is loaded
		auto chunk_loader = [this](const std::pair<int, vec2>& local_chunk, const cudaArray_t loading_dest[2]) -> bool {
			//Load the texture to the given array.
			//If texture is not loaded the given array will be cleared automatically
			return this->MapLoader(local_chunk.second, loading_dest);
		};

		//launching async loading
		const int original_size = loading_chunks.size();
		auto current_node = loading_chunks.begin();
		auto heightfield_node = heightfield.begin();
		for (int threadID = 0; threadID < original_size; threadID++) {
			//start loading
			loader.emplace(this->compute_pool->enqueue_future(chunk_loader, *current_node, (*heightfield_node).get()));

			heightfield_node++;
			current_node++;
		}

		//waiting for loading
		//get the result
		current_node = loading_chunks.begin();
		for (int i = 0; i < original_size; i++) {
			if (loader.front().get()) {
				//loaded
				//traversing the list while erasing it is quite good	
				current_node = loading_chunks.erase(current_node);
				//the current_node will be pointing to the next node automatically
			}
			else {
				//not yet loaded, computation has been dispatched, so we keep this chunk
				current_node++;
				//it won't go beyong the end pointer
			}

			//delete the mapped array
			loader.pop();
		}

		//release the mapping
		heightfield.clear();
		//deleted by smart ptr
		return loading_chunks.size();
	};

	//texture storage
	const int loading_size = loading_chunks.size();
	static std::list<std::unique_ptr<cudaArray_t[]>> heightfield;
	//map the texture, all opengl related work must be done on the main contexted thread
	cudaGraphicsMapResources(2, this->heightfield_texture_res);
	//get the mapped array on the main contexte thread
	auto node = loading_chunks.begin();
	for (int i = 0; i < loading_size; i++) {
		//allocating spaces for this chunk (2 tetxures per chunk)
		std::unique_ptr<cudaArray_t[]> heightfield_ptr = std::unique_ptr<cudaArray_t[]>(new cudaArray_t[2]);
		//map each texture
		cudaGraphicsSubResourceGetMappedArray(&(heightfield_ptr[0]), this->heightfield_texture_res[0], node->first, 0);
		cudaGraphicsSubResourceGetMappedArray(&(heightfield_ptr[1]), this->heightfield_texture_res[1], node->first, 0);
		heightfield.push_back(std::move(heightfield_ptr));
		node++;
	}
	//start loading chunk
	this->ChunkLoader = this->compute_pool->enqueue_future(asyncChunkLoader, std::ref(heightfield));
	return true;
}

bool STPChunkManager::loadChunksAsync(vec3 cameraPos) {
	const STPSettings::STPChunkSettings* chunk_settings = this->ChunkProvider.getChunkSettings();
	//waiting for the previous worker to finish(if any)
	this->SyncloadChunks();
	//make sure there isn't any worker accessing the loading_chunks, otherwise undefined behaviour warning

	//check if the central position has changed or not
	const vec2 thisCentralPos = STPChunk::getChunkPosition(cameraPos - chunk_settings->ChunkOffset, chunk_settings->ChunkSize, chunk_settings->ChunkScaling);
	if (thisCentralPos != this->lastCentralPos) {
		//changed
		//recalculate loading chunks
		this->loadingLocals.clear();
		const auto allChunks = STPChunk::getRegion(
			STPChunk::getChunkPosition(
				cameraPos - chunk_settings->ChunkOffset,
				chunk_settings->ChunkSize,
				chunk_settings->ChunkScaling),
			chunk_settings->ChunkSize,
			chunk_settings->RenderedChunk,
			chunk_settings->ChunkScaling);
		
		//we also need chunkID, which is just the index of the visible chunk from top-left to bottom-right
		int chunkID = 0;
		for (auto it = allChunks.begin(); it != allChunks.end(); it++) {
			this->loadingLocals.push_back(std::pair<int, vec2>(chunkID, *it));
			chunkID++;
		}

		this->lastCentralPos = thisCentralPos;
	}

	return this->loadChunksAsync(this->loadingLocals);
}

int STPChunkManager::SyncloadChunks() {
	//sync the chunk loading and make sure the chunk loader has finished before return
	if (this->ChunkLoader.valid()) {
		//unmap the chunk first
		cudaGraphicsUnmapResources(2, this->heightfield_texture_res);

		return this->ChunkLoader.get();
	}
	else {
		//chunk loader hasn't started
		return -1;
	}
}

STPChunkProvider& STPChunkManager::getChunkProvider() {
	return this->ChunkProvider;
}
#pragma warning(default : 4267)