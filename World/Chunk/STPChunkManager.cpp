#pragma warning(disable : 4267)//implicit conversion may lose data wwarning (actually it won't)
#include "STPChunkManager.h"

using glm::vec2;
using glm::vec4;
using glm::value_ptr;

using std::list;
using std::queue;
using std::unique_ptr;
using std::pair;
using std::make_pair;

using namespace SuperTerrainPlus;

STPChunkManager::STPChunkManager(STPSettings::STPConfigurations* settings) : ChunkCache(), ChunkProvider(settings) {

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

	//create thread pool
	//TODO: data racing identified, multithreading disabled for now. Change it back to 5u after it's fixed
	this->compute_pool = std::unique_ptr<STPThreadPool>(new STPThreadPool(2u));
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

bool STPChunkManager::loadMap(vec2 chunkPos, const cudaArray_t destination[2]) {
	const STPSettings::STPChunkSettings* chunk_settings = this->ChunkProvider.getChunkSettings();

	//ask provider if we can get the chunk
	STPChunk* chunk = this->ChunkProvider.requestChunk(this->ChunkCache, chunkPos);

	if (chunk != nullptr) {
		//chunk is ready, we can start loading
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
	auto asyncChunkLoader = [this, &loading_chunks](list<pair<vec2, unique_ptr<cudaArray_t[]>>>& chunk_data) -> unsigned int {
		queue<std::future<bool>, list<std::future<bool>>> loading_status;
		//A functoin to load one chunk
		//return true if chunk is loaded
		auto chunk_loader = [this](vec2 local_chunk, const cudaArray_t loading_dest[2]) -> bool {
			//Load the texture to the given array.
			//If texture is not loaded the given array will be cleared automatically
			return this->loadMap(local_chunk, loading_dest);
		};
		
		//EXPERIMENTAL FEATURE BEGINS
		//a function to compute chunk to make sure its available when loading 
		auto chunk_computer = [this](vec2 local_chunk) -> bool{
			return this->ChunkProvider.checkChunk(this->ChunkCache, local_chunk);
		};
		//launching async computing
		for (auto& local : chunk_data) {
			loading_status.emplace(this->compute_pool->enqueue_future(chunk_computer, local.first));
		}
		//waiting for compute to finish before loading
		while (!loading_status.empty()) {
			//it actually doesn't matter if it returns false
			loading_status.front().get();

			//delete
			loading_status.pop();
		}
		//EXPERIMENTAL FEATURE ENDS

		//launching async loading
		for (auto it = chunk_data.begin(); it != chunk_data.end(); it++) {
			//start loading
			loading_status.emplace(this->compute_pool->enqueue_future(chunk_loader, it->first, it->second.get()));
		}

		//waiting for loading
		//get the result
		for (auto it = loading_chunks.begin(); it != loading_chunks.end();) {
			if (loading_status.front().get()) {
				//loaded
				//traversing the list while erasing it is quite good	
				it = loading_chunks.erase(it);
				//the current_node will be pointing to the next node automatically
			}
			else {
				//not yet loaded, computation has been dispatched, so we keep this chunk
				it++;
				//it won't go beyong the end pointer
			}

			//delete the future
			loading_status.pop();
		}

		//release the mapping
		chunk_data.clear();
		//deleted by smart ptr
		return loading_chunks.size();
	};

	//texture storage
	//map the texture, all opengl related work must be done on the main contexted thread
	cudaGraphicsMapResources(2, this->heightfield_texture_res);
	//get the mapped array on the main contexte thread
	auto node = loading_chunks.begin();
	for (auto node = loading_chunks.begin(); node != loading_chunks.end(); node++) {
		//allocating spaces for this chunk (2 tetxures per chunk)
		unique_ptr<cudaArray_t[]> heightfield_ptr(new cudaArray_t[2]);
		//map each texture
		cudaGraphicsSubResourceGetMappedArray(&(heightfield_ptr[0]), this->heightfield_texture_res[0], node->first, 0);
		cudaGraphicsSubResourceGetMappedArray(&(heightfield_ptr[1]), this->heightfield_texture_res[1], node->first, 0);
		this->chunk_data.emplace_back(node->second, std::move(heightfield_ptr));
	}
	//start loading chunk
	this->ChunkLoader = this->compute_pool->enqueue_future(asyncChunkLoader, std::ref(this->chunk_data));
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
		//wait for finish first
		const unsigned int res = this->ChunkLoader.get();
		//unmap the chunk
		cudaGraphicsUnmapResources(2, this->heightfield_texture_res);

		return static_cast<int>(res);
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