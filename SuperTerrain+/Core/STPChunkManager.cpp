#pragma warning(disable : 4267)//implicit conversion may lose data warning (actually it won't)
#include <SuperTerrain+/World/Chunk/STPChunkManager.h>

#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

using glm::uvec2;
using glm::ivec2;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::value_ptr;

using std::queue;
using std::bind;
using std::unique_ptr;
using std::make_unique;
using std::pair;
using std::future;
using namespace std::placeholders;
using std::make_pair;

using namespace SuperTerrainPlus;

STPChunkManager::STPChunkManager(STPChunkProvider& provider) : ChunkProvider(provider), trigger_clearBuffer(false), compute_pool(1u), buffering_stream(cudaStreamNonBlocking) {
	const STPEnvironment::STPChunkSetting& chunk_setting = this->ChunkProvider.getChunkSetting();
	const ivec2 buffer_size(chunk_setting.RenderedChunk * chunk_setting.MapSize);
	const int totaltexture_size = buffer_size.x * buffer_size.y * sizeof(unsigned short) * 4;//4 channel

	//creating texture
	glCreateTextures(GL_TEXTURE_2D, 2, this->terrain_heightfield);
	//allocate for biomemap
	glTextureParameteri(this->terrain_heightfield[0], GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(this->terrain_heightfield[0], GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTextureParameteri(this->terrain_heightfield[0], GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTextureParameteri(this->terrain_heightfield[0], GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTextureParameteri(this->terrain_heightfield[0], GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//R format
	glTextureStorage2D(this->terrain_heightfield[0], 1, GL_R16UI, buffer_size.x, buffer_size.y);
	STPcudaCheckErr(cudaGraphicsGLRegisterImage(this->heightfield_texture_res, this->terrain_heightfield[0], GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

	//allocate space for heightfield
	glTextureParameteri(this->terrain_heightfield[1], GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(this->terrain_heightfield[1], GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTextureParameteri(this->terrain_heightfield[1], GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTextureParameteri(this->terrain_heightfield[1], GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTextureParameteri(this->terrain_heightfield[1], GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//RGBA format
	glTextureStorage2D(this->terrain_heightfield[1], 1, GL_RGBA16, buffer_size.x, buffer_size.y);
	STPcudaCheckErr(cudaGraphicsGLRegisterImage(this->heightfield_texture_res + 1, this->terrain_heightfield[1], GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

	//init clear buffers that are used to clear texture when new rendered chunks are loaded (we need to clear the previous chunk data)
	STPcudaCheckErr(cudaMallocHost(&this->quad_clear, totaltexture_size));
	memset(this->quad_clear, 0x88, totaltexture_size);

	this->renderingLocals.reserve(chunk_setting.RenderedChunk.x * chunk_setting.RenderedChunk.y);
}

STPChunkManager::~STPChunkManager() {
	//wait until the worker has finished
	if (this->ChunkLoader.valid()) {
		this->ChunkLoader.get();
	}

	//wait for the stream to finish
	STPcudaCheckErr(cudaStreamSynchronize(this->buffering_stream));

	//unregister resource
	STPcudaCheckErr(cudaGraphicsUnregisterResource(this->heightfield_texture_res[0]));
	STPcudaCheckErr(cudaGraphicsUnregisterResource(this->heightfield_texture_res[1]));
	//delete texture
	glDeleteTextures(2, this->terrain_heightfield);

	//delete clear buffer
	STPcudaCheckErr(cudaFreeHost(this->quad_clear));
}

bool STPChunkManager::renderingBufferChunkSubData(cudaArray_t buffer[2], vec2 chunkPos, unsigned int chunkID) {
	const STPEnvironment::STPChunkSetting& chunk_setting = this->ChunkProvider.getChunkSetting();
	//ask provider if we can get the chunk
	STPChunk* chunk = this->ChunkProvider.requestChunk(chunkPos);
	if (chunk == nullptr) {
		//not ready yet
		return false;
	}

	//chunk is ready, copy to rendering buffer
	const uvec2& rendered_chunk = chunk_setting.RenderedChunk,
		& dimension = chunk_setting.MapSize;
	auto calcBufferOffset = [&rendered_chunk](unsigned int chunkID, const uvec2& dimension) -> uvec2 {
		//calculate global offset, basically
		const uvec2 chunkIdx(chunkID % rendered_chunk.x, chunkID / rendered_chunk.x);
		return dimension * chunkIdx;
	};
	const uvec2 buffer_offset = calcBufferOffset(chunkID, dimension);

	size_t pixel_size;
	//copy biomemap
	pixel_size = sizeof(STPDiversity::Sample);
	STPcudaCheckErr(cudaMemcpy2DToArrayAsync(buffer[0], buffer_offset.x * pixel_size, buffer_offset.y, chunk->getBiomemap(),
		dimension.x * pixel_size, dimension.x * pixel_size,
		dimension.y, cudaMemcpyHostToDevice, this->buffering_stream));
	//copy heightfield
	pixel_size = sizeof(unsigned short) * 4;
	STPcudaCheckErr(cudaMemcpy2DToArrayAsync(buffer[1], buffer_offset.x * pixel_size, buffer_offset.y, chunk->getRenderingBuffer(),
		dimension.x * pixel_size, dimension.x * pixel_size,
		dimension.y, cudaMemcpyHostToDevice, this->buffering_stream));

	return true;
}

void STPChunkManager::clearRenderingBuffer(cudaArray_t destination, size_t pixel_size) {
	const STPEnvironment::STPChunkSetting& chunk_setting = this->ChunkProvider.getChunkSetting();
	const ivec2 buffer_size(chunk_setting.RenderedChunk * chunk_setting.MapSize);

	//clear unloaded chunk, so the engine won't display the chunk from previous rendered chunks
	STPcudaCheckErr(cudaMemcpy2DToArrayAsync(destination, 0, 0, this->quad_clear,
		buffer_size.x * pixel_size, buffer_size.x * pixel_size,
		buffer_size.y, cudaMemcpyHostToDevice, this->buffering_stream));
	STPcudaCheckErr(cudaStreamSynchronize(this->buffering_stream));
}

void STPChunkManager::generateMipmaps() {
	glGenerateTextureMipmap(this->terrain_heightfield[1]);
}

bool STPChunkManager::loadChunksAsync(STPLocalChunks& loading_chunks) {
	//waiting for the previous worker to finish(if any)
	//make sure there isn't any worker accessing the loading_chunks, otherwise undefined behaviour warning
	this->SyncloadChunks();
	if (loading_chunks.size() == 0) {
		return false;
	}

	//async chunk loader
	auto asyncChunkLoader = [this, &loading_chunks](cudaArray_t biomemap_data, cudaArray_t heightfield_data) -> unsigned int {
		//requesting rendering buffer
		unsigned int num_chunkLoaded = 0u;
		for (int i = 0; i < loading_chunks.size(); i++) {
			auto& [chunkPos, chunkLoaded] = loading_chunks[i];

			if (chunkLoaded) {
				//skip this chunk if loading has been completed before
				continue;
			}
			//make sure chunk is available, if not we need to compute it
			if (!this->getChunkProvider().checkChunk(chunkPos, bind(&STPChunkManager::reloadChunkAsync, this, _1))) {
				//chunk is in used, skip it for now
				continue;
			}
			
			//load chunk into rendering buffer
			cudaArray_t chunk_data[2] = { biomemap_data, heightfield_data };
			if (this->renderingBufferChunkSubData(chunk_data, chunkPos, i)) {
				//loaded
				chunkLoaded = true;
				num_chunkLoaded++;
			}
		}
		//wait until everything is finished
		STPcudaCheckErr(cudaStreamSynchronize(this->buffering_stream));
		
		return num_chunkLoaded;
	};

	//texture storage
	//map the texture, all opengl related work must be done on the main contexted thread
	STPcudaCheckErr(cudaGraphicsMapResources(2, this->heightfield_texture_res));
	cudaArray_t biomemap_ptr, heightfield_ptr;
	//we only have one texture, so index is always zero
	STPcudaCheckErr(cudaGraphicsSubResourceGetMappedArray(&biomemap_ptr, this->heightfield_texture_res[0], 0, 0));
	STPcudaCheckErr(cudaGraphicsSubResourceGetMappedArray(&heightfield_ptr, this->heightfield_texture_res[1], 0, 0));
	//clear up the render buffer for every chunk
	if (this->trigger_clearBuffer) {
		this->clearRenderingBuffer(biomemap_ptr, sizeof(STPDiversity::Sample));
		this->clearRenderingBuffer(heightfield_ptr, sizeof(unsigned short) * 4);
		this->trigger_clearBuffer = false;
	}

	//start loading chunk
	this->ChunkLoader = this->compute_pool.enqueue_future(asyncChunkLoader, biomemap_ptr, heightfield_ptr);
	return true;
}

bool STPChunkManager::loadChunksAsync(vec3 cameraPos) {
	const STPEnvironment::STPChunkSetting& chunk_setting = this->ChunkProvider.getChunkSetting();
	//waiting for the previous worker to finish(if any)
	this->SyncloadChunks();
	//make sure there isn't any worker accessing the loading_chunks, otherwise undefined behaviour warning

	//check if the central position has changed or not
	if (const vec2 thisCentralPos = STPChunk::getChunkPosition(cameraPos - chunk_setting.ChunkOffset, chunk_setting.ChunkSize, chunk_setting.ChunkScaling); 
		thisCentralPos != this->lastCentralPos) {
		//changed
		//recalculate loading chunks
		this->renderingLocals.clear();
		this->renderingLocals_lookup.clear();
		const auto allChunks = STPChunk::getRegion(
			STPChunk::getChunkPosition(
				cameraPos - chunk_setting.ChunkOffset,
				chunk_setting.ChunkSize,
				chunk_setting.ChunkScaling),
			chunk_setting.ChunkSize,
			chunk_setting.RenderedChunk,
			chunk_setting.ChunkScaling);
		
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
		STPcudaCheckErr(cudaGraphicsUnmapResources(2, this->heightfield_texture_res));
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

GLuint STPChunkManager::getCurrentRenderingBuffer(STPRenderingBufferType type) const {
	unsigned int index;

	switch (type) {
	case STPRenderingBufferType::BIOME: index = 0u;
		break;
	case STPRenderingBufferType::HEIGHTFIELD: index = 1u;
		break;
	default:
		//impossible
		break;
	}
	return this->terrain_heightfield[index];
}
#pragma warning(default : 4267)