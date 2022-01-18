#include <SuperTerrain+/World/STPWorldPipeline.h>

//Error Handling
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Exception/STPAsyncGenerationError.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>

//GL
#include <glad/glad.h>
//CUDA
#include <cuda_gl_interop.h>

#include <algorithm>
#include <sstream>
#include <type_traits>

//GLM
using glm::ivec2;
using glm::uvec2;
using glm::vec2;
using glm::ivec3;
using glm::vec3;

using std::vector;
using std::list;
using std::queue;
using std::shared_mutex;
using std::shared_lock;
using std::unique_lock;
using std::exception_ptr;
using std::make_pair;

using std::for_each;
using std::make_unique;

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;
using namespace SuperTerrainPlus::STPCompute;

//Used by STPGeneratorManager for storing async exceptions
#define STORE_EXCEPTION(FUN) try { \
	FUN; \
} \
catch (...) { \
	{ \
		unique_lock<shared_mutex> newExceptionLock(this->ExceptionStorageLock); \
		this->ExceptionStorage.emplace(std::current_exception()); \
	} \
}

#define BUFFER_INDEX(M) static_cast<std::underlying_type_t<STPWorldPipeline::STPRenderingBufferType>>(M)
#define FOR_EACH_BUFFER(FUN) for(unsigned int i = 0u; i < STPWorldPipeline::BufferCount; i++) { \
	FUN; \
}

class STPWorldPipeline::STPGeneratorManager {
private:

	STPThreadPool GeneratorWorker;

	STPChunkStorage ChunkCache;

	//all terrain map generators
	STPBiomeFactory& generateBiomemap;
	STPHeightfieldGenerator& generateHeightfield;

public:

	STPTextureFactory& generateSplatmap;

private:

	STPWorldPipeline* const Pipeline;

	//Store exception thrown from async execution
	queue<exception_ptr> ExceptionStorage;
	shared_mutex ExceptionStorageLock;

	//Contains pointers of chunk.
	typedef list<STPChunk*> STPChunkRecord;

	/**
	 * @brief Calculate the chunk offset such that the transition of each chunk is seamless
	 * @param chunkPos The world position of the chunk
	 * @return The chunk offset in world coordinate.
	*/
	inline vec2 calcOffset(vec2 chunkPos) const {
		const STPEnvironment::STPChunkSetting& chk_config = this->ChunkSetting;
		return STPChunk::calcChunkMapOffset(chunkPos, chk_config.ChunkSize, chk_config.MapSize, chk_config.MapOffset, chk_config.ChunkScaling);
	}

	/**
	 * @brief Get all neighbour for this chunk position
	 * @param chunkPos The central chunk world position
	 * @return A list of all neighbour chunk position
	*/
	inline STPChunk::STPChunkPositionCache getNeighbour(vec2 chunkPos) const {
		const STPEnvironment::STPChunkSetting& chk_config = this->ChunkSetting;
		return STPChunk::getRegion(chunkPos, chk_config.ChunkSize, chk_config.FreeSlipChunk, chk_config.ChunkScaling);
	}

	/**
	 * @brief Dispatch compute for heightmap, the heightmap result will be writen back to the storage
	 * @param current_chunk The maps for the chunk
	 * @param neighbour_chunk The maps of the chunks that require to be used for biome-edge interpolation during heightmap generation,
	 * require the central chunk and neighbour chunks arranged in row-major flavour. The central chunk should also be included.
	 * @param chunkPos The world position of the chunk
	*/
	void computeHeightmap(STPChunk* current_chunk, STPChunkRecord& neighbour_chunk, vec2 chunkPos) {
		//generate heightmap
		STPHeightfieldGenerator::STPMapStorage maps;
		maps.Biomemap.reserve(neighbour_chunk.size());
		maps.Heightmap32F.reserve(1ull);
		for (STPChunk* chk : neighbour_chunk) {
			maps.Biomemap.push_back(chk->getBiomemap());
		}
		maps.Heightmap32F.push_back(current_chunk->getHeightmap());
		maps.HeightmapOffset = this->calcOffset(chunkPos);
		const STPHeightfieldGenerator::STPGeneratorOperation op = 
			STPHeightfieldGenerator::HeightmapGeneration;

		//computing heightmap
		this->generateHeightfield(maps, op);
	}

	/**
	 * @brief Dispatch compute for free-slip hydraulic erosion, normalmap compute and formatting, requires heightmap presenting in the chunk
	 * @param neighbour_chunk The maps of the chunks that require to be eroded with a free-slip manner, require the central chunk and neighbour chunks
	 * arranged in row-major flavour. The central chunk should also be included.
	*/
	void computeErosion(STPChunkRecord& neighbour_chunk) {
		STPHeightfieldGenerator::STPMapStorage maps;
		maps.Heightmap32F.reserve(neighbour_chunk.size());
		maps.Heightfield16UI.reserve(neighbour_chunk.size());
		for (STPChunk* chk : neighbour_chunk) {
			maps.Heightmap32F.push_back(chk->getHeightmap());
			maps.Heightfield16UI.push_back(chk->getRenderingBuffer());
		}
		const STPHeightfieldGenerator::STPGeneratorOperation op =
			STPHeightfieldGenerator::Erosion |
			STPHeightfieldGenerator::RenderingBufferGeneration;

		//computing and return success state
		this->generateHeightfield(maps, op);
	}

	/**
	 * @brief Recursively prepare neighbour chunks for the central chunk.
	 * The first recursion will prepare neighbour biomemaps for heightmap generation.
	 * The second recursion will prepare neighbour heightmaps for erosion.
	 * @param chunkPos The position to the chunk which should be prepared.
	 * @param rec_depth Please leave this empty, this is the recursion depth and will be managed properly
	 * @return If all neighbours are ready to be used, true is returned.
	 * If any neighbour is not ready (being used by other threads or neighbour is not ready and compute is launched), return false
	*/
	const STPChunk* recNeighbourChecking(vec2 chunkPos, unsigned char rec_depth = 2u) {
		//recursive case:
		//define what rec_depth means...
		constexpr static unsigned char BIOMEMAP_PASS = 1u,
			HEIGHTMAP_PASS = 2u;

		{
			STPChunk::STPChunkState expected_state = STPChunk::STPChunkState::Empty;
			switch (rec_depth) {
			case BIOMEMAP_PASS: expected_state = STPChunk::STPChunkState::Heightmap_Ready;
				break;
			case HEIGHTMAP_PASS: expected_state = STPChunk::STPChunkState::Complete;
				break;
			default:
				break;
			}
			if (STPChunk* center = this->ChunkCache[chunkPos];
				center != nullptr && center->getChunkState() >= expected_state) {
				//no need to continue if center chunk is available
				//since the center chunk might be used as a neighbour chunk later, we only return bool instead of a pointer
				//after checkChunk() is performed for every chunks, we can grab all pointers and check for occupancy in other functions.
				return center;
			}
		}
		auto biomemap_computer = [this](STPChunk* chunk, vec2 offset) -> void {
			//since biomemap is discrete, we need to round the pixel
			ivec2 rounded_offset = static_cast<ivec2>(glm::round(offset));
			STORE_EXCEPTION(this->generateBiomemap(chunk->getBiomemap(), ivec3(rounded_offset.x, 0, rounded_offset.y)))
				//computation was successful
				chunk->markChunkState(STPChunk::STPChunkState::Biomemap_Ready);
			chunk->markOccupancy(false);
		};
		auto heightmap_computer = [this](STPChunk* chunk, STPChunkRecord neighbours, vec2 position) -> void {
			STORE_EXCEPTION(this->computeHeightmap(chunk, neighbours, position))
				//computation was successful
				chunk->markChunkState(STPChunk::STPChunkState::Heightmap_Ready);
			//unlock all neighbours
			for_each(neighbours.begin(), neighbours.end(), [](auto c) -> void { c->markOccupancy(false); });
		};
		auto erosion_computer = [this](STPChunk* centre, STPChunkRecord neighbours) -> void {
			STORE_EXCEPTION(this->computeErosion(neighbours))
				//erosion was successful
				//mark center chunk complete
				centre->markChunkState(STPChunk::STPChunkState::Complete);
			for_each(neighbours.begin(), neighbours.end(), [](auto c) -> void { c->markOccupancy(false); });
		};

		//reminder: central chunk is included in neighbours
		const STPEnvironment::STPChunkSetting& chk_config = this->ChunkSetting;
		const STPChunk::STPChunkPositionCache neighbour_position = this->getNeighbour(chunkPos);

		bool canContinue = true;
		//The first pass: check if all neighbours are ready for some operations
		STPChunkRecord neighbour;
		for (vec2 neighbourPos : neighbour_position) {
			//get current neighbour chunk
			STPChunkStorage::STPChunkConstructed res = this->ChunkCache.construct(neighbourPos, chk_config.MapSize);
			STPChunk* curr_neighbour = res.second;

			if (curr_neighbour->isOccupied()) {
				//occupied means it's currently in used (probably another thread has already started to compute it)
				canContinue = false;
				continue;
			}
			switch (rec_depth) {
			case BIOMEMAP_PASS:
				//container will guaranteed to exists since heightmap pass has already created it
				if (curr_neighbour->getChunkState() == STPChunk::STPChunkState::Empty) {
					curr_neighbour->markOccupancy(true);
					//compute biomemap
					this->GeneratorWorker.enqueue_void(biomemap_computer, curr_neighbour, this->calcOffset(neighbourPos));
					//try to compute all biomemap, and when biomemap is computing, we don't need to wait
					canContinue = false;
				}
				break;
			case HEIGHTMAP_PASS:
				//check neighbouring biomemap
				if (!this->recNeighbourChecking(neighbourPos, rec_depth - 1u)) {
					canContinue = false;
				}
				break;
			default:
				//never gonna happen
				break;
			}

			neighbour.push_back(curr_neighbour);
			//if chunk is found, we can guarantee it's in-used empty or at least biomemap/heightmap complete
		}
		if (!canContinue) {
			//if biomemap/heightmap is computing, we don't need to check for heightmap generation/erosion because some chunks are in use
			return nullptr;
		}

		//The second pass: launch compute on the center with all neighbours
		if (std::any_of(neighbour.begin(), neighbour.end(), [](auto c) -> bool { return c->isOccupied(); })) {
			//if any of the chunk is occupied, we cannot continue
			return nullptr;
		}
		//all chunks are available, lock all neighbours
		for_each(neighbour.begin(), neighbour.end(), [](auto c) -> void { c->markOccupancy(true); });
		//send the list of neighbour chunks to GPU to perform some operations
		switch (rec_depth) {
		case BIOMEMAP_PASS:
			//generate heightmap
			this->GeneratorWorker.enqueue_void(heightmap_computer, this->ChunkCache[chunkPos], neighbour, chunkPos);
			break;
		case HEIGHTMAP_PASS:
			//perform erosion on heightmap
			this->GeneratorWorker.enqueue_void(erosion_computer, this->ChunkCache[chunkPos], neighbour);
			{
				//trigger a chunk reload as some chunks have been added to render buffer already after neighbours are updated
				for (vec2 position : this->getNeighbour(chunkPos)) {
					this->Pipeline->reload(position);
				}
			}
			break;
		default:
			//never gonna happen
			break;
		}

		//compute has been launched
		return nullptr;
	}

public:

	const STPEnvironment::STPChunkSetting& ChunkSetting;

	/**
	 * @brief Intialise generator manager with pipeline stages filled with generators.
	 * @param setup The pointer to all pipeline stages.
	 * @param pipeline The pointer to the world pipeline registered with the generator manager.
	*/
	STPGeneratorManager(STPWorldPipeline::STPPipelineSetup& setup, STPWorldPipeline* pipeline) : 
		generateBiomemap(*setup.BiomemapGenerator), generateHeightfield(*setup.HeightfieldGenerator), generateSplatmap(*setup.SplatmapGenerator), 
		ChunkSetting(*setup.ChunkSetting), Pipeline(pipeline), GeneratorWorker(5u) {

	}

	STPGeneratorManager(const STPGeneratorManager&) = delete;

	STPGeneratorManager(STPGeneratorManager&&) = delete;

	STPGeneratorManager& operator=(const STPGeneratorManager&) = delete;

	STPGeneratorManager& operator=(STPGeneratorManager&&) = delete;

	~STPGeneratorManager() = default;

	/**
	 * @brief Prepare and generate splatmap for rendering.
	 * @param buffer Rendering buffer on device side, a mapped OpenGL pointer.
	 * @param requesting_chunk Specify chunks that need to have splatmap generated.
	 * Note that the coordinate of local chunk should specify chunk map offset rather than chunk world position.
	 * @param stream The CUDA stream where work will be submitted to.
	*/
	void computeSplatmap(const STPRenderingBufferMemory& buffer, const STPTextureFactory::STPRequestingChunkInfo& requesting_chunk, cudaStream_t stream) {
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
		tex_desc.normalizedCoords = 0;

		//biomemap
		res_desc.res.array.array = buffer.Map[BUFFER_INDEX(STPWorldPipeline::STPRenderingBufferType::BIOME)];
		tex_desc.filterMode = cudaFilterModePoint;
		tex_desc.readMode = cudaReadModeElementType;
		STPcudaCheckErr(cudaCreateTextureObject(&biomemap, &res_desc, &tex_desc, nullptr));

		//heightfield
		res_desc.res.array.array = buffer.Map[BUFFER_INDEX(STPWorldPipeline::STPRenderingBufferType::HEIGHTFIELD)];
		tex_desc.filterMode = cudaFilterModeLinear;
		tex_desc.readMode = cudaReadModeNormalizedFloat;
		STPcudaCheckErr(cudaCreateTextureObject(&heightfield, &res_desc, &tex_desc, nullptr));

		//splatmap
		res_desc.res.array.array = buffer.Map[BUFFER_INDEX(STPWorldPipeline::STPRenderingBufferType::SPLAT)];
		STPcudaCheckErr(cudaCreateSurfaceObject(&splatmap, &res_desc));

		//launch splatmap computer
		this->generateSplatmap(biomemap, heightfield, splatmap, requesting_chunk, stream);

		//before deallocation happens make sure everything has finished.
		STPcudaCheckErr(cudaStreamSynchronize(stream));
		//finish up, we have to delete it everytime because resource array may change every time we re-map it
		STPcudaCheckErr(cudaDestroyTextureObject(biomemap));
		STPcudaCheckErr(cudaDestroyTextureObject(heightfield));
		STPcudaCheckErr(cudaDestroySurfaceObject(splatmap));
	}

	/**
	 * @brief Request a pointer to the chunk given a world coordinate.
	 * @param world_coord The world position where the chunk is requesting.
	 * @return The pointer to the requsted chunk.
	 * The function returns a valid point only when the chunk is fully ready for rendering.
	 * In case chunk is not ready, such as being used by other chunks, or map generation is in progress, nullptr is returned.
	*/
	const STPChunk* getChunk(vec2 world_coord) {
		//check if there's any exception thrown from previous async compute launch
		bool hasException;
		{
			shared_lock<shared_mutex> checkExceptionLock(this->ExceptionStorageLock);
			hasException = !this->ExceptionStorage.empty();
		}
		if (hasException) {
			unique_lock<shared_mutex> clearExceptionLock(this->ExceptionStorageLock);
			std::stringstream error_message;

			//merge all exception messages
			while (!this->ExceptionStorage.empty()) {
				std::exception_ptr exptr = this->ExceptionStorage.front();
				this->ExceptionStorage.pop();
				try {
					std::rethrow_exception(exptr);
				}
				catch (const std::exception& e) {
					//unfortunately we will lose all exception type information :(
					error_message << e.what() << std::endl;
				}
			}
			//throw the compound exception out
			throw STPException::STPAsyncGenerationError(error_message.str().c_str());
		}

		//recursively preparing neighbours
		return this->recNeighbourChecking(world_coord);
	}

};

STPWorldPipeline::STPWorldPipeline(STPPipelineSetup& setup) : PipelineWorker(1u), Generator(make_unique<STPGeneratorManager>(setup, this)), 
	BufferStream(cudaStreamNonBlocking), ChunkSetting(*setup.ChunkSetting) {
	const STPEnvironment::STPChunkSetting& setting = this->ChunkSetting;
	const uvec2 buffer_size(setting.RenderedChunk * setting.MapSize);
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
	glCreateTextures(GL_TEXTURE_2D, 3, this->TerrainMap);
	//biomemap in R format
	setupTex(this->TerrainMap[0], GL_NEAREST, GL_NEAREST, 1, GL_R16UI);
	regTex(this->TerrainMap[0], this->TerrainMapRes);
	//heightfield in R format
	setupTex(this->TerrainMap[1], GL_LINEAR, GL_LINEAR, 1, GL_R16);
	regTex(this->TerrainMap[1], this->TerrainMapRes + 1);
	//splatmap in R format
	setupTex(this->TerrainMap[2], GL_NEAREST, GL_NEAREST, 1, GL_R8UI);
	regTex(this->TerrainMap[2], this->TerrainMapRes + 2, cudaGraphicsRegisterFlagsSurfaceLoadStore);

	//init clear buffers that are used to clear texture when new rendered chunks are loaded (we need to clear the previous chunk data)
	const uvec2 clearBuffer_format = buffer_size * uvec2(sizeof(unsigned short), 1u);
	STPcudaCheckErr(cudaMallocPitch(&this->TerrainMapClearBuffer, &this->TerrainMapClearBufferPitch, clearBuffer_format.x, clearBuffer_format.y));
	STPcudaCheckErr(cudaMemset2D(this->TerrainMapClearBuffer, this->TerrainMapClearBufferPitch, 0x88u, clearBuffer_format.x, clearBuffer_format.y));

	//init rendering back buffer cache
	auto allocate_cache = [&buffer_size](size_t& pitch, size_t channelSize) -> void* {
		void* mem;
		STPcudaCheckErr(cudaMallocPitch(&mem, &pitch, buffer_size.x * channelSize, buffer_size.y));
		return mem;
	};
	FOR_EACH_BUFFER(this->TerrainMapExchangeCache.MapCache[i] = allocate_cache(this->TerrainMapExchangeCache.Pitch[i], STPRenderingBufferMemory::Format[i]))

	const unsigned int renderedChunkCount = setting.RenderedChunk.x * setting.RenderedChunk.y;
	this->renderingLocal.reserve(renderedChunkCount);
	this->renderingLocalLookup.reserve(renderedChunkCount);
	this->TerrainMapExchangeCache.LocalCache.reserve(renderedChunkCount);
}

STPWorldPipeline::~STPWorldPipeline() {
	//wait until the worker has finished
	if (this->MapLoader.valid()) {
		this->MapLoader.get();
	}

	//wait for the stream to finish
	STPcudaCheckErr(cudaStreamSynchronize(*this->BufferStream));

	//unregister resource
	for (int i = 0; i < 3; i++) {
		STPcudaCheckErr(cudaGraphicsUnregisterResource(this->TerrainMapRes[i]));
	}
	//delete texture
	glDeleteTextures(3, this->TerrainMap);

	//delete clear buffer
	STPcudaCheckErr(cudaFree(this->TerrainMapClearBuffer));
	//delete rendering back buffer cache
	FOR_EACH_BUFFER(cudaFree(this->TerrainMapExchangeCache.MapCache[i]))
}

void STPWorldPipeline::copySubBufferFromSubCache
	(const STPRenderingBufferMemory& dest, unsigned int dest_idx, unsigned int src_idx) {
	auto copy_from_cache = [stream = *this->BufferStream, &dimension = this->ChunkSetting.MapSize,
		src_offset = this->calcBufferOffset(src_idx), dest_offset = this->calcBufferOffset(dest_idx)]
		(cudaArray_t dest, const void* src, size_t src_pitch, size_t channelSize) -> void {
		//calculate the first pixel to be copied from
		const unsigned char* src_start = 
			(reinterpret_cast<const unsigned char*>(src) + src_offset.y * src_pitch) + src_offset.x * channelSize;

		STPcudaCheckErr(cudaMemcpy2DToArrayAsync(dest, dest_offset.x * channelSize, dest_offset.y,
			src_start, src_pitch,
			dimension.x * channelSize, dimension.y, cudaMemcpyDeviceToDevice, stream));
	};

	//copy old cache to the new buffer
	FOR_EACH_BUFFER(copy_from_cache(dest.Map[i], 
		this->TerrainMapExchangeCache.MapCache[i], this->TerrainMapExchangeCache.Pitch[i], STPRenderingBufferMemory::Format[i]))
}

void STPWorldPipeline::backupBuffer(const STPRenderingBufferMemory& buffer) {
	const STPEnvironment::STPChunkSetting& chunk_setting = this->ChunkSetting;
	auto copy_buffer = [stream = *this->BufferStream, buffer_size = chunk_setting.RenderedChunk * chunk_setting.MapSize]
		(void* dest, size_t dest_pitch, cudaArray_t src, size_t channelSize) -> void {
		STPcudaCheckErr(cudaMemcpy2DFromArrayAsync(dest, dest_pitch, src, 0, 0, buffer_size.x * channelSize, buffer_size.y, cudaMemcpyDeviceToDevice, stream));
	};

	//backup our rendering
	FOR_EACH_BUFFER(copy_buffer(this->TerrainMapExchangeCache.MapCache[i], this->TerrainMapExchangeCache.Pitch[i], 
		buffer.Map[i], STPRenderingBufferMemory::Format[i]))
}

void STPWorldPipeline::clearBuffer(const STPRenderingBufferMemory& destination, unsigned int dest_idx) {
	auto erase_buffer = [stream = *this->BufferStream, clearBuffer = this->TerrainMapClearBuffer, clearBufferPitch = this->TerrainMapClearBufferPitch,
		&dimension = this->ChunkSetting.MapSize, buffer_offset = this->calcBufferOffset(dest_idx)](cudaArray_t dest, size_t channelSize) -> void {
		STPcudaCheckErr(cudaMemcpy2DToArrayAsync(dest, buffer_offset.x * channelSize, buffer_offset.y, 
			clearBuffer, clearBufferPitch, 
			dimension.x * channelSize, dimension.y, cudaMemcpyDeviceToDevice, stream));
	};

	//clear unloaded chunk, so the engine won't display the chunk from previous rendered chunks
	FOR_EACH_BUFFER(erase_buffer(destination.Map[i], STPRenderingBufferMemory::Format[i]))
}

bool STPWorldPipeline::mapSubData(const STPRenderingBufferMemory& buffer, vec2 chunkPos, unsigned int chunkID) {
	//ask provider if we can get the chunk
	const STPChunk* chunk = this->Generator->getChunk(chunkPos);
	if (chunk == nullptr) {
		//not ready yet
		return false;
	}

	//chunk is ready, copy to rendering buffer
	auto copy_buffer = [stream = *this->BufferStream, &dimension = this->ChunkSetting.MapSize, buffer_offset = this->calcBufferOffset(chunkID)]
		(cudaArray_t dest, const void* src, size_t channelSize) -> void {
		STPcudaCheckErr(cudaMemcpy2DToArrayAsync(dest, buffer_offset.x * channelSize, buffer_offset.y, src,
			dimension.x * channelSize, dimension.x * channelSize,
			dimension.y, cudaMemcpyHostToDevice, stream));
	};

	unsigned int index;
	//copy buffer to GL texture
	index = BUFFER_INDEX(STPWorldPipeline::STPRenderingBufferType::BIOME);
	copy_buffer(buffer.Map[index], chunk->getBiomemap(), STPRenderingBufferMemory::Format[index]);
	index = BUFFER_INDEX(STPWorldPipeline::STPRenderingBufferType::HEIGHTFIELD);
	copy_buffer(buffer.Map[index], chunk->getRenderingBuffer(), STPRenderingBufferMemory::Format[index]);

	return true;
}

uvec2 STPWorldPipeline::calcBufferOffset(unsigned int index) const {
	const STPEnvironment::STPChunkSetting& chunk_setting = this->ChunkSetting;
	const unsigned int rendered_width = chunk_setting.RenderedChunk.x;

	//calculate global offset, basically
	const uvec2 chunkIdx(index % rendered_width, index / rendered_width);
	return chunk_setting.MapSize * chunkIdx;
}

bool STPWorldPipeline::load(const vec3& cameraPos) {
	const STPEnvironment::STPChunkSetting& chunk_setting = this->ChunkSetting;
	//waiting for the previous worker to finish(if any)
	this->wait();
	//make sure there isn't any worker accessing the loading_chunks, otherwise undefined behaviour warning

	//Whenever camera changes location, all previous rendering buffers are dumpped
	bool centreChanged = false, 
		shouldClearBuffer = false;
	//check if the central position has changed or not
	if (const vec2 thisCentralPos = STPChunk::getChunkPosition(cameraPos - chunk_setting.ChunkOffset, chunk_setting.ChunkSize, chunk_setting.ChunkScaling);
		thisCentralPos != this->lastCenterLocation) {
		//changed
		centreChanged = true;
		//backup the current rendering locals
		this->TerrainMapExchangeCache.LocalCache.clear();
		for (auto [local_it, chunkID] = make_pair(this->renderingLocal.cbegin(), 0u); local_it != this->renderingLocal.cend(); local_it++, chunkID++) {
			const auto [position, status] = *local_it;
			this->TerrainMapExchangeCache.LocalCache.try_emplace(position, chunkID, status);
		}

		//recalculate loading chunks
		this->renderingLocal.clear();
		this->renderingLocalLookup.clear();
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
			this->renderingLocal.emplace_back(*it, false);
			this->renderingLocalLookup.emplace(*it, chunkID);
		}

		this->lastCenterLocation = thisCentralPos;
		//clear up previous rendering buffer
		shouldClearBuffer = true;
	}
	else if (std::all_of(this->renderingLocal.cbegin(), this->renderingLocal.cend(), [](auto i) -> bool {return i.second; })) {
		//if all chunks are loaded there is no need to do those complicated stuff
		return false;
	}

	//otherwise start loading
	if (this->renderingLocal.size() == 0) {
		throw STPException::STPMemoryError("Unable to retrieve a list of chunks need to be rendered.");
	}
	//async chunk loader
	auto asyncChunkLoader = [this, &chunk_setting](STPRenderingBufferMemory map_data) -> void {
		//requesting rendering buffer
		unsigned int num_chunkLoaded = 0u;
		//keep a record of which chunk's rendering buffer has been updated
		STPDiversity::STPTextureFactory::STPRequestingChunkInfo updated_chunk;
		for (unsigned int i = 0u; i < this->renderingLocal.size(); i++) {
			auto& [chunkPos, chunkLoaded] = this->renderingLocal[i];

			if (chunkLoaded) {
				//skip this chunk if loading has been completed before
				continue;
			}
			//load chunk into rendering buffer
			if (this->mapSubData(map_data, chunkPos, i)) {
				//loaded
				chunkLoaded = true;
				num_chunkLoaded++;
				//mark updated rendering buffer
				//we need to use the chunk normalised coordinate to get the splatmap offset, 
				//splatmap offset needs to be consistent with the heightmap and biomemap
				const vec2 offset = STPChunk::calcChunkMapOffset(
					chunkPos,
					chunk_setting.ChunkSize,
					chunk_setting.MapSize,
					chunk_setting.MapOffset,
					chunk_setting.ChunkScaling);
				//local chunk coordinate
				const uvec2 local_coord = STPChunk::getLocalChunkCoordinate(i, chunk_setting.RenderedChunk);
				updated_chunk.emplace_back(STPDiversity::STPTextureInformation::STPSplatGeneratorInformation::STPLocalChunkInformation
					{ local_coord.x, local_coord.y, offset.x, offset.y });
			}
		}
		if (!updated_chunk.empty()) {
			//there exists chunk that has rendering buffer updated, we need to update splatmap as well
			this->Generator->computeSplatmap(map_data, updated_chunk, *this->BufferStream);
		}
	};

	//texture storage
	//map the texture, all opengl related work must be done on the main contexted thread
	//CUDA will make sure all previous graphics API calls are finished before stream begins
	STPcudaCheckErr(cudaGraphicsMapResources(3, this->TerrainMapRes, *this->BufferStream));
	STPRenderingBufferMemory buffer_ptr;
	//we only have one texture, so index is always zero
	FOR_EACH_BUFFER(STPcudaCheckErr(cudaGraphicsSubResourceGetMappedArray(buffer_ptr.Map + i, this->TerrainMapRes[i], 0, 0)))
	//clear up the render buffer for every chunk
	if (shouldClearBuffer) {
		//backup the current buffer before clearing
		this->backupBuffer(buffer_ptr);

		const STPLocalChunkCache& cacheDict = this->TerrainMapExchangeCache.LocalCache;
		//here we perform an optimisation: reuse chunk that has been rendered previously from the old rendering buffer
		for (auto& [chunkPos, loaded] : this->renderingLocal) {
			//check in the new rendered chunk, is there any old chunk has the same world coordinate as the new chunk?
			auto pos_it = cacheDict.find(chunkPos);

			const unsigned int buffer_chunkID = this->renderingLocalLookup.at(chunkPos);
			if (pos_it != cacheDict.cend()) {
				//found, check if the previous cache is complete

				const auto [cache_chunkID, cache_status] = pos_it->second;
				if (cache_status) {
					//if the previous cahce is complete, copy to new buffer
					this->copySubBufferFromSubCache(buffer_ptr, buffer_chunkID, cache_chunkID);
					//mark this chunk as loaded
					loaded = true;

					continue;
				}

			}
			//the current new chunk has no usesable buffer, we need to load it from chunk storage later
			//clear this chunk
			this->clearBuffer(buffer_ptr, buffer_chunkID);

		}
	}

	//group mapped data together and start loading chunk
	this->MapLoader = this->PipelineWorker.enqueue_future(asyncChunkLoader, buffer_ptr);
	return centreChanged;
}

bool STPWorldPipeline::reload(const vec2& chunkPos) {
	auto it = this->renderingLocalLookup.find(chunkPos);
	if (it == this->renderingLocalLookup.end()) {
		//chunk position provided is not required to be rendered, or new rendering area has changed
		return false;
	}
	//found, trigger a reload
	this->renderingLocal[it->second].second = false;
	return true;
}

void STPWorldPipeline::wait() {
	//sync the chunk loading and make sure the chunk loader has finished before return
	if (this->MapLoader.valid()) {
		//wait for finish first
		this->MapLoader.get();
		//sync the stream that modifies the texture and unmap the chunk
		//CUDA will make sure all previous pending works in the stream has finished before graphics API can be called
		STPcudaCheckErr(cudaGraphicsUnmapResources(3, this->TerrainMapRes, *this->BufferStream));
	}
}

const vec2& STPWorldPipeline::centre() const {
	return this->lastCenterLocation;
}

STPOpenGL::STPuint STPWorldPipeline::operator[](STPRenderingBufferType type) const {
	return this->TerrainMap[static_cast<std::underlying_type_t<STPRenderingBufferType>>(type)];
}

STPDiversity::STPTextureFactory& STPWorldPipeline::splatmapGenerator() const {
	return this->Generator->generateSplatmap;
}