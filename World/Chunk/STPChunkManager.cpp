#pragma warning(disable : 4267)//implicit conversion may lose data wwarning (actually it won't)
#include "STPChunkManager.h"

using glm::ivec2;
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

STPChunkManager::STPChunkManager(STPSettings::STPConfigurations* settings) : ChunkProvider(settings) {
	const STPSettings::STPChunkSettings* chunk_settings = this->ChunkProvider.getChunkSettings();
	const ivec2 buffer_size(chunk_settings->RenderedChunk * chunk_settings->MapSize);
	const int totaltexture_size = buffer_size.x * buffer_size.y * sizeof(unsigned short) * 4;//4 channel

	//creating texture
	glCreateTextures(GL_TEXTURE_2D, 1, &this->terrain_heightfield);
	//and allocating spaces for heightfield
	glTextureParameteri(this->terrain_heightfield, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(this->terrain_heightfield, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTextureParameteri(this->terrain_heightfield, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTextureParameteri(this->terrain_heightfield, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTextureParameteri(this->terrain_heightfield, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//RGBA format
	glTextureStorage2D(this->terrain_heightfield, 1, GL_RGBA16, buffer_size.x, buffer_size.y);
	cudaGraphicsGLRegisterImage(&this->heightfield_texture_res, this->terrain_heightfield, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

	//init clear buffers that are used to clear texture when new rendered chunks are loaded (we need to clear the previous chunk data)
	cudaMallocHost(&this->quad_clear, totaltexture_size);
	memset(this->quad_clear, 0x88, totaltexture_size);

	this->renderingLocals.reserve(chunk_settings->RenderedChunk.x * chunk_settings->RenderedChunk.y);
	//create thread pool
	this->compute_pool = make_unique<STPThreadPool>(1u);
	//create stream
	cudaStreamCreateWithFlags(&this->buffering_stream, cudaStreamNonBlocking);
}

STPChunkManager::~STPChunkManager() {
	//wait until the worker has finished
	if (this->ChunkLoader.valid()) {
		this->ChunkLoader.get();
	}
	//wait until stream has finished
	cudaStreamSynchronize(this->buffering_stream);
	cudaStreamDestroy(this->buffering_stream);

	//unregister resource
	cudaGraphicsUnregisterResource(this->heightfield_texture_res);
	//delete texture
	glDeleteTextures(1, &this->terrain_heightfield);

	//delete clear buffer
	cudaFreeHost(this->quad_clear);
}

bool STPChunkManager::renderingBufferSubData(cudaArray_t buffer, vec2 chunkPos, unsigned int chunkID) {
	const STPSettings::STPChunkSettings* chunk_settings = this->ChunkProvider.getChunkSettings();
	//ask provider if we can get the chunk
	STPChunk* chunk = this->ChunkProvider.requestChunk(chunkPos);
	if (chunk == nullptr) {
		//not ready yet
		return false;
	}

	//chunk is ready, copy to rendering buffer
	const uvec2& rendered_chunk = this->getChunkProvider().getChunkSettings()->RenderedChunk,
		& dimension = this->getChunkProvider().getChunkSettings()->MapSize;
	auto calcBufferOffset = [&rendered_chunk](unsigned int chunkID, const uvec2& dimension) -> uvec2 {
		//calculate global offset, basically
		const uvec2 chunkIdx(chunkID % rendered_chunk.x, static_cast<unsigned int>(floorf(1.0f * chunkID / rendered_chunk.x)));
		return uvec2(dimension.x * chunkIdx.x, dimension.y * chunkIdx.y);
	};
	const uvec2 buffer_offset = calcBufferOffset(chunkID, dimension);
	return cudaSuccess == cudaMemcpy2DToArrayAsync(buffer, buffer_offset.x * sizeof(unsigned short) * 4, buffer_offset.y, chunk->getRenderingBuffer(),
		dimension.x * sizeof(unsigned short) * 4, dimension.x * sizeof(unsigned short) * 4,
		dimension.y, cudaMemcpyHostToDevice, this->buffering_stream);
}

void STPChunkManager::clearRenderingBuffer(cudaArray_t destination) {
	const STPSettings::STPChunkSettings* chunk_settings = this->ChunkProvider.getChunkSettings();
	const ivec2 buffer_size(chunk_settings->RenderedChunk * chunk_settings->MapSize);

	//clear unloaded chunk, so the engine won't display the chunk from previous rendered chunks
	cudaMemcpy2DToArrayAsync(destination, 0, 0, this->quad_clear,
		buffer_size.x * sizeof(unsigned short) * 4, buffer_size.x * sizeof(unsigned short) * 4,
		buffer_size.y, cudaMemcpyHostToDevice, this->buffering_stream);
	cudaStreamSynchronize(this->buffering_stream);
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
	auto asyncChunkLoader = [this, &loading_chunks](cudaArray_t chunk_data) -> unsigned int {
		//requesting rendering buffer
		unsigned int num_chunkLoaded = 0u;
		for (int i = 0; i < loading_chunks.size(); i++) {
			auto& current_chunk = loading_chunks[i];

			if (current_chunk.second) {
				//skip this chunk if loading has been completed before
				continue;
			}
			//make sure chunk is available, if not we need to compute it
			if (!this->getChunkProvider().checkChunk(current_chunk.first, bind(&STPChunkManager::reloadChunkAsync, this, _1))) {
				//chunk is in used, skip it for now
				continue;
			}
			
			//load chunk into rendering buffer
			if (this->renderingBufferSubData(chunk_data, current_chunk.first, i)) {
				//loaded
				num_chunkLoaded++;
			}
		}
		//wait until everything is finished
		cudaStreamSynchronize(this->buffering_stream);
		
		return num_chunkLoaded;
	};

	//texture storage
	//map the texture, all opengl related work must be done on the main contexted thread
	cudaGraphicsMapResources(1, &this->heightfield_texture_res);
	cudaArray_t heightfield_ptr;
	//we only have one texture, so index is always zero
	cudaGraphicsSubResourceGetMappedArray(&heightfield_ptr, this->heightfield_texture_res, 0, 0);
	//clear up the render buffer for every chunk
	if (this->trigger_clearBuffer) {
		this->clearRenderingBuffer(heightfield_ptr);
		this->trigger_clearBuffer = false;
	}

	//start loading chunk
	this->ChunkLoader = this->compute_pool->enqueue_future(asyncChunkLoader, heightfield_ptr);
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