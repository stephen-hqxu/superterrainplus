#include <SuperTerrain+/World/Chunk/STPChunkManager.h>

#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

//GLAD
#include <glad/glad.h>
//CUDA + GL
#include <cuda_gl_interop.h>

#include <type_traits>

using glm::uvec2;
using glm::ivec2;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::value_ptr;

using std::vector;
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
	auto setupTex = [buffer_size](GLuint texture, GLint min_filter, GLint mag_filter, GLsizei levels, GLenum internalFormat) -> void {
		//set texture parameter
		glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTextureParameteri(texture, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
		glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, min_filter);
		glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, mag_filter);
		//allocation
		glTextureStorage2D(texture, levels, internalFormat, buffer_size.x, buffer_size.y);
	};
	auto regTex = [](GLuint texture, cudaGraphicsResource_t* res, unsigned int reg_flag = cudaGraphicsRegisterFlagsNone) -> void {
		//register cuda texture
		STPcudaCheckErr(cudaGraphicsGLRegisterImage(res, texture, GL_TEXTURE_2D, reg_flag));
		STPcudaCheckErr(cudaGraphicsResourceSetMapFlags(*res, cudaGraphicsMapFlagsNone));
	};

	//creating texture
	glCreateTextures(GL_TEXTURE_2D, 3, this->terrain_heightfield);
	//biomemap in R format
	setupTex(this->terrain_heightfield[0], GL_NEAREST, GL_NEAREST, 1, GL_R16UI);
	regTex(this->terrain_heightfield[0], this->heightfield_texture_res);
	//heightfield in R format
	setupTex(this->terrain_heightfield[1], GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, 1, GL_R16);
	regTex(this->terrain_heightfield[1], this->heightfield_texture_res + 1);
	//splatmap in R format
	setupTex(this->terrain_heightfield[2], GL_NEAREST, GL_NEAREST, 1, GL_R8UI);
	regTex(this->terrain_heightfield[2], this->heightfield_texture_res + 2, cudaGraphicsRegisterFlagsSurfaceLoadStore);

	//init clear buffers that are used to clear texture when new rendered chunks are loaded (we need to clear the previous chunk data)
	const int totaltexture_size = buffer_size.x * buffer_size.y * sizeof(unsigned short);//1 channel
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
	for (int i = 0; i < 3; i++) {
		STPcudaCheckErr(cudaGraphicsUnregisterResource(this->heightfield_texture_res[i]));
	}
	//delete texture
	glDeleteTextures(3, this->terrain_heightfield);

	//delete clear buffer
	STPcudaCheckErr(cudaFreeHost(this->quad_clear));
}

bool STPChunkManager::renderingBufferChunkSubData(const STPRenderingBufferMemory& buffer, vec2 chunkPos, unsigned int chunkID) {
	const STPEnvironment::STPChunkSetting& chunk_setting = this->ChunkProvider.getChunkSetting();
	//ask provider if we can get the chunk
	STPChunk* chunk = this->ChunkProvider.requestChunk(chunkPos);
	if (chunk == nullptr) {
		//not ready yet
		return false;
	}

	//chunk is ready, copy to rendering buffer
	const uvec2& rendered_chunk = chunk_setting.RenderedChunk,
		 &dimension = chunk_setting.MapSize;
	auto calcBufferOffset = [&rendered_chunk](unsigned int chunkID, const uvec2& dimension) -> uvec2 {
		//calculate global offset, basically
		const uvec2 chunkIdx(chunkID % rendered_chunk.x, chunkID / rendered_chunk.x);
		return dimension * chunkIdx;
	};
	auto copy_buffer = [this, &dimension, buffer_offset = calcBufferOffset(chunkID, dimension)](cudaArray_t dest, const void* src, size_t channelSize) -> void {
		STPcudaCheckErr(cudaMemcpy2DToArrayAsync(dest, buffer_offset.x * channelSize, buffer_offset.y, src,
			dimension.x * channelSize, dimension.x * channelSize,
			dimension.y, cudaMemcpyHostToDevice, this->buffering_stream));
	};

	//copy buffer to GL texture
	copy_buffer(buffer.Biomemap, chunk->getBiomemap(), STPRenderingBufferMemory::BiomemapChannel);
	copy_buffer(buffer.Heightfield, chunk->getRenderingBuffer(), STPRenderingBufferMemory::HeightfieldChannel);

	return true;
}

void STPChunkManager::prepareSplatmap(const STPRenderingBufferMemory& buffer, const STPDiversity::STPTextureFactory::STPRequestingChunkInfo& requesting_chunk) {
	//prepare texture and surface object
	cudaTextureObject_t biomemap, heightfield;
	cudaSurfaceObject_t splatmap;
	//make sure all description are zero init
	cudaResourceDesc res_desc = { };
	res_desc.resType = cudaResourceTypeArray;
	cudaTextureDesc tex_desc = { };
	tex_desc.addressMode[0] = cudaAddressModeClamp;
	tex_desc.addressMode[1] = cudaAddressModeClamp;
	tex_desc.addressMode[2] = cudaAddressModeClamp;

	//biomemap
	res_desc.res.array.array = buffer.Biomemap;
	tex_desc.filterMode = cudaFilterModePoint;
	tex_desc.readMode = cudaReadModeElementType;
	STPcudaCheckErr(cudaCreateTextureObject(&biomemap, &res_desc, &tex_desc, nullptr));

	//heightfield
	res_desc.res.array.array = buffer.Heightfield;
	tex_desc.filterMode = cudaFilterModeLinear;
	tex_desc.readMode = cudaReadModeNormalizedFloat;
	STPcudaCheckErr(cudaCreateTextureObject(&heightfield, &res_desc, &tex_desc, nullptr));

	//splatmap
	res_desc.res.array.array = buffer.Splatmap;
	STPcudaCheckErr(cudaCreateSurfaceObject(&splatmap, &res_desc));

	//TODO: launch splatmap computer

	//finish up, we have to delete it everytime because resource array may change every time we re-map it
	STPcudaCheckErr(cudaDestroyTextureObject(biomemap));
	STPcudaCheckErr(cudaDestroyTextureObject(heightfield));
	STPcudaCheckErr(cudaDestroySurfaceObject(splatmap));
}

void STPChunkManager::clearRenderingBuffer(const STPRenderingBufferMemory& destination) {
	const STPEnvironment::STPChunkSetting& chunk_setting = this->ChunkProvider.getChunkSetting();
	auto erase_buffer = [this, buffer_size = chunk_setting.RenderedChunk * chunk_setting.MapSize](cudaArray_t data, size_t channelSize) -> void {
		STPcudaCheckErr(cudaMemcpy2DToArrayAsync(data, 0, 0, this->quad_clear,
			buffer_size.x * channelSize, buffer_size.x * channelSize,
			buffer_size.y, cudaMemcpyHostToDevice, this->buffering_stream));
	};

	//clear unloaded chunk, so the engine won't display the chunk from previous rendered chunks
	erase_buffer(destination.Biomemap, STPRenderingBufferMemory::BiomemapChannel);
	erase_buffer(destination.Heightfield, STPRenderingBufferMemory::HeightfieldChannel);
	erase_buffer(destination.Splatmap, STPRenderingBufferMemory::SplatmapChannel);
}

void STPChunkManager::generateMipmaps() {
	glGenerateTextureMipmap(this->terrain_heightfield[1]);
}

bool STPChunkManager::loadChunksAsync(STPLocalChunkStatus& loading_chunks) {
	//waiting for the previous worker to finish(if any)
	//make sure there isn't any worker accessing the loading_chunks, otherwise undefined behaviour warning
	this->SyncloadChunks();
	if (loading_chunks.size() == 0) {
		return false;
	}

	//async chunk loader
	auto asyncChunkLoader = [this, &loading_chunks](STPRenderingBufferMemory map_data) -> unsigned int {
		const STPEnvironment::STPChunkSetting& chunk_setting = this->ChunkProvider.getChunkSetting();
		//requesting rendering buffer
		unsigned int num_chunkLoaded = 0u;
		//keep a record of which chunk's rendering buffer has been updated
		STPDiversity::STPTextureFactory::STPRequestingChunkInfo updated_chunk;
		for (unsigned int i = 0u; i < loading_chunks.size(); i++) {
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
			if (this->renderingBufferChunkSubData(map_data, chunkPos, i)) {
				//loaded
				chunkLoaded = true;
				num_chunkLoaded++;
				//mark updated rendering buffer
				//we need to use the chunk normalised coordinate to get the splatmap offset, splatmap offset needs to be consistent with the heightmap and biomemap
				updated_chunk.emplace_back(STPDiversity::STPTextureFactory::STPLocalChunkInformation{
					i,
					STPChunk::calcChunkMapOffset(
					chunkPos,
					chunk_setting.ChunkSize,
					chunk_setting.MapSize,
					chunk_setting.MapOffset,
					chunk_setting.ChunkScaling)
				});
			}
		}
		if (!updated_chunk.empty()) {
			//there exists chunk that has rendering buffer updated, we need to update splatmap as well
			//TODO: enable splatmap generation
			//this->prepareSplatmap(map_data, updated_chunk);
		}
		
		return num_chunkLoaded;
	};

	//texture storage
	//map the texture, all opengl related work must be done on the main contexted thread
	STPcudaCheckErr(cudaGraphicsMapResources(3, this->heightfield_texture_res, this->buffering_stream));
	STPRenderingBufferMemory buffer_ptr;
	//we only have one texture, so index is always zero
	STPcudaCheckErr(cudaGraphicsSubResourceGetMappedArray(&buffer_ptr.Biomemap, this->heightfield_texture_res[0], 0, 0));
	STPcudaCheckErr(cudaGraphicsSubResourceGetMappedArray(&buffer_ptr.Heightfield, this->heightfield_texture_res[1], 0, 0));
	STPcudaCheckErr(cudaGraphicsSubResourceGetMappedArray(&buffer_ptr.Splatmap, this->heightfield_texture_res[2], 0, 0));
	//clear up the render buffer for every chunk
	if (this->trigger_clearBuffer) {
		this->clearRenderingBuffer(buffer_ptr);
		this->trigger_clearBuffer = false;
	}

	//group mapped data together and start loading chunk
	this->ChunkLoader = this->compute_pool.enqueue_future(asyncChunkLoader, buffer_ptr);
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
		for (auto [it, chunkID] = make_pair(allChunks.begin(), 0u); it != allChunks.end(); it++, chunkID++) {
			this->renderingLocals.emplace_back(*it, false);
			this->renderingLocals_lookup.emplace(*it, chunkID);
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
		//sync the stream that modifies the texture and unmap the chunk
		STPcudaCheckErr(cudaGraphicsUnmapResources(3, this->heightfield_texture_res, this->buffering_stream));
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
	return this->terrain_heightfield[static_cast<std::underlying_type_t<STPRenderingBufferType>>(type)];
}