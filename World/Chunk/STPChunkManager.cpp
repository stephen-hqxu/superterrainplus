#pragma warning(disable : 4267)//implicit conversion may lose data wwarning (actually it won't)
#include "STPChunkManager.h"

using glm::vec2;
using glm::vec4;
using glm::value_ptr;

using std::list;
using std::queue;
using std::bind;
using std::unique_ptr;
using std::make_unique;
using std::pair;
using std::future;
using namespace std::placeholders;
using std::make_pair;

using namespace SuperTerrainPlus;

STPChunkManager::STPChunkManager(STPSettings::STPConfigurations* settings) : ChunkCache(), ChunkProvider(settings) {
	const STPSettings::STPChunkSettings* chunk_settings = this->ChunkProvider.getChunkSettings();
	const int chunk_num = static_cast<int>(chunk_settings->RenderedChunk.x * chunk_settings->RenderedChunk.y);
	const int totaltexture_size = chunk_num * static_cast<int>(chunk_settings->MapSize.x * chunk_settings->MapSize.y) * sizeof(unsigned short);//one channel

	//creating texture
	glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, &this->terrain_heightfield);
	//and allocating spaces for heightfield
	glTextureParameteri(this->terrain_heightfield, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(this->terrain_heightfield, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTextureParameteri(this->terrain_heightfield, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTextureParameteri(this->terrain_heightfield, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTextureParameteri(this->terrain_heightfield, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//RGBA format
	glTextureStorage3D(this->terrain_heightfield, 1, GL_RGBA16, static_cast<int>(chunk_settings->MapSize.x), static_cast<int>(chunk_settings->MapSize.y), chunk_num);
	cudaGraphicsGLRegisterImage(&this->heightfield_texture_res, this->terrain_heightfield, GL_TEXTURE_2D_ARRAY, cudaGraphicsRegisterFlagsWriteDiscard);

	//init clear buffers that are used to clear texture when new rendered chunks are loaded (we need to clear the previous chunk data)
	cudaMallocHost(&this->quad_clear, totaltexture_size * 4);
	memset(this->quad_clear, 0x88, totaltexture_size * 4);

	this->renderingLocals.reserve(chunk_settings->RenderedChunk.x * chunk_settings->RenderedChunk.y);
	//create thread pool
	this->compute_pool = make_unique<STPThreadPool>(4u);
}

STPChunkManager::~STPChunkManager() {
	//wait until the worker has finished
	if (this->ChunkLoader.valid()) {
		this->ChunkLoader.get();
	}

	//unregister resource
	cudaGraphicsUnregisterResource(this->heightfield_texture_res);
	//delete texture
	glDeleteTextures(1, &this->terrain_heightfield);

	//delete clear buffer
	cudaFreeHost(this->quad_clear);
}

bool STPChunkManager::loadRenderingBuffer(vec2 chunkPos, const cudaArray_t destination) {
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
		no_error &= cudaSuccess == cudaMemcpy2DToArrayAsync(destination, 0, 0, chunk->getRenderingBuffer(),
			static_cast<int>(chunk->getSize().x) * sizeof(unsigned short) * 4, static_cast<int>(chunk->getSize().x) * sizeof(unsigned short) * 4,
			static_cast<int>(chunk->getSize().y), cudaMemcpyHostToDevice, copy_stream);
		no_error &= cudaSuccess == cudaStreamSynchronize(copy_stream);
		no_error &= cudaSuccess == cudaStreamDestroy(copy_stream);

		return no_error;
	}

	//not ready yet
	return false;
}

void STPChunkManager::clearRenderingBuffer(const cudaArray_t destination) {
	const STPSettings::STPChunkSettings* chunk_settings = this->ChunkProvider.getChunkSettings();
	//clear unloaded chunk, so the engine won't display the chunk from previous rendered chunks
	cudaStream_t clear_stream;
	cudaStreamCreateWithFlags(&clear_stream, cudaStreamNonBlocking);
	cudaMemcpy2DToArrayAsync(destination, 0, 0, this->quad_clear,
		static_cast<int>(chunk_settings->MapSize.x) * sizeof(unsigned short) * 4, static_cast<int>(chunk_settings->MapSize.x) * sizeof(unsigned short) * 4,
		static_cast<int>(chunk_settings->MapSize.y), cudaMemcpyHostToDevice, clear_stream);
	cudaStreamSynchronize(clear_stream);
	cudaStreamDestroy(clear_stream);
}

void STPChunkManager::generateMipmaps() {
	glGenerateTextureMipmap(this->terrain_heightfield);
}

bool STPChunkManager::loadChunksAsync(STPLocalChunks& loading_chunks) {
	//waiting for the previous worker to finish(if any)
	//make sure there isn't any worker accessing the loading_chunks, otherwise undefined behaviour warning
	this->SyncloadChunks();
	if (loading_chunks.size() == 0) {
		return false;
	}

	//async chunk loader
	auto asyncChunkLoader = [this, &loading_chunks](list<pair<vec2, cudaArray_t>>& chunk_data) -> unsigned int {
		//chunk future being loading and the chunk ID
		list<pair<future<bool>, int>> loading_status;
		
		//make sure chunk is available, if not we need to compute it
		for (auto& local : chunk_data) {
			this->getChunkProvider().checkChunk(this->ChunkCache, local.first, bind(&STPChunkManager::reloadChunkAsync, this, _1));
		}

		//launching async loading
		int i = 0;
		for (auto it = chunk_data.begin(); it != chunk_data.end(); it++) {
			//load the chunk into rendering buffer only when the chunk is no loaded yet
			if (!loading_chunks[i].second) {
				loading_status.emplace_back(this->compute_pool->enqueue_future(bind(&STPChunkManager::loadRenderingBuffer, this, _1, _2), it->first, it->second), i);
			}
			i++;
		}

		//waiting for loading
		//get the result
		unsigned int num_chunkLoaded = 0u;
		for (auto it = loading_status.begin(); it != loading_status.end(); it = loading_status.erase(it)) {
			auto& current_chunk = loading_chunks[it->second];
			if (it->first.get()) {
				//loaded
				current_chunk.second = true;
				num_chunkLoaded++;
			}
			//delete the future in the next iteration
		}

		//mapped pointers will be released when SyncloadChunks() is called
		chunk_data.clear();
		return num_chunkLoaded;
	};

	//texture storage
	//map the texture, all opengl related work must be done on the main contexted thread
	cudaGraphicsMapResources(1, &this->heightfield_texture_res);
	//get the mapped array on the main contexte thread
	for (int i = 0; i < loading_chunks.size(); i++) {
		auto node = loading_chunks[i];
		//allocating spaces for this chunk (2 tetxures per chunk)
		cudaArray_t heightfield_ptr;
		//map each texture
		cudaGraphicsSubResourceGetMappedArray(&heightfield_ptr, this->heightfield_texture_res, i, 0);
		//clear up the render buffer for every chunk
		if (this->trigger_clearBuffer) {
			this->clearRenderingBuffer(heightfield_ptr);
		}

		this->chunk_data.emplace_back(node.first, heightfield_ptr);
	}
	this->trigger_clearBuffer = false;
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
		this->renderingLocals.clear();
		this->renderingLocals_lookup.clear();
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
			this->renderingLocals.emplace_back(*it, false);
			this->renderingLocals_lookup.emplace(*it, chunkID++);
		}

		this->lastCentralPos = thisCentralPos;
		//clear up previous rendering buffer
		this->trigger_clearBuffer = true;
	}
	else if (std::all_of(this->renderingLocals.cbegin(), this->renderingLocals.cend(), [](auto i) -> bool {return i.second;})) {
		//if all chunks are loaded there is no need to do those complicated stuff
		return false;
	}

	return this->loadChunksAsync(this->renderingLocals);
}

bool STPChunkManager::reloadChunkAsync(vec2 chunkPos) {
	auto it = this->renderingLocals_lookup.find(chunkPos);
	if (it == this->renderingLocals_lookup.end()) {
		//chunk position provided is not required to be rendered, or new rendering area has changed
		return false;
	}
	//found, trigger a reload
	this->renderingLocals[it->second].second = false;
	return true;
}

int STPChunkManager::SyncloadChunks() {
	//sync the chunk loading and make sure the chunk loader has finished before return
	if (this->ChunkLoader.valid()) {
		//wait for finish first
		const unsigned int res = this->ChunkLoader.get();
		//unmap the chunk
		if (cudaGraphicsUnmapResources(1, &this->heightfield_texture_res) != cudaSuccess) {
			throw std::runtime_error("CUDA resource cannot be released");
		}

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