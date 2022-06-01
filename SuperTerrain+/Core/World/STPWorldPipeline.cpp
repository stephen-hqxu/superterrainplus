#include <SuperTerrain+/World/STPWorldPipeline.h>

//Error Handling
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Exception/STPAsyncGenerationError.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>

//Hasher
#include <SuperTerrain+/Utility/STPHashCombine.h>

//GL
#include <glad/glad.h>
//CUDA
#include <cuda_gl_interop.h>

#include <glm/common.hpp>

//Container
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <list>
#include <queue>
#include <optional>

#include <limits>
#include <algorithm>
#include <sstream>
#include <shared_mutex>
#include <type_traits>

//GLM
using glm::ivec2;
using glm::uvec2;
using glm::vec2;
using glm::dvec2;
using glm::ivec3;
using glm::dvec3;

using std::array;
using std::list;
using std::optional;
using std::pair;
using std::queue;
using std::unordered_map;
using std::unordered_set;
using std::vector;

using std::atomic_bool;
using std::exception_ptr;
using std::future;
using std::mutex;
using std::shared_lock;
using std::shared_mutex;
using std::unique_lock;

using std::for_each;
using std::make_pair;
using std::make_optional;
using std::make_unique;
using std::nullopt;

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;

//Used by STPGeneratorManager for storing asynchronous exceptions
#define STORE_EXCEPTION(FUN) try { \
	FUN; \
} \
catch (...) { \
	{ \
		unique_lock<shared_mutex> newExceptionLock(this->ExceptionStorageLock); \
		this->ExceptionStorage.emplace(std::current_exception()); \
	} \
}

#define TERRAIN_MAP_INDEX(M) static_cast<std::underlying_type_t<STPWorldPipeline::STPTerrainMapType>>(M)

/**
 * @brief The hash function for the glm::ivec2
 */
struct STPHashivec2 {
public:

	inline size_t operator()(const ivec2& position) const {
		//combine hash
		return STPHashCombine::combine(0ull, position.x, position.y);
	}

};

class STPWorldPipeline::STPGeneratorManager {
private:

	unordered_map<ivec2, STPChunk, STPHashivec2> ChunkCache;

	//all terrain map generators
	STPBiomeFactory& generateBiomemap;
	STPHeightfieldGenerator& generateHeightfield;

public:

	STPTextureFactory& generateSplatmap;

private:

	STPWorldPipeline& Pipeline;
	const STPEnvironment::STPChunkSetting& ChunkSetting;

	//Store exception thrown from asynchronous execution
	queue<exception_ptr> ExceptionStorage;
	shared_mutex ExceptionStorageLock;

	//Contains pointers of chunk.
	typedef vector<STPChunk*> STPChunkRecord;
	//Contains visitors of chunk which guarantees unique access.
	typedef vector<STPChunk::STPUniqueMapVisitor> STPUniqueChunkRecord;

	typedef list<STPUniqueChunkRecord> STPUniqueChunkCache_t;
	//A temporary storage of unique chunk map visitor to be passed to other threads.
	//For data safety, remove the iterator from the container while accessing the underlying value;
	STPUniqueChunkCache_t UniqueChunkCache;
	mutex UniqueChunkCacheLock;
	typedef STPUniqueChunkCache_t::iterator STPUniqueChunkCacheEntry;

	STPThreadPool GeneratorWorker;

	/**
	 * @brief Calculate the chunk offset such that the transition of each chunk is seamless
	 * @param chunkCoord The world coordinate of the chunk
	 * @return The chunk offset in world coordinate.
	*/
	inline dvec2 calcOffset(const ivec2& chunkCoord) const noexcept {
		const STPEnvironment::STPChunkSetting& chk_config = this->ChunkSetting;
		return STPChunk::calcChunkMapOffset(chunkCoord, chk_config.ChunkSize, chk_config.MapSize, chk_config.MapOffset);
	}

	/**
	 * @brief Get all neighbour for this chunk position
	 * @param chunkCoord The central chunk world coordinate
	 * @return A list of all neighbour chunk position
	*/
	inline STPChunk::STPChunkCoordinateCache getNeighbour(const ivec2& chunkCoord) const noexcept {
		const STPEnvironment::STPChunkSetting& chk_config = this->ChunkSetting;
		return STPChunk::calcChunkNeighbour(chunkCoord, chk_config.ChunkSize, chk_config.FreeSlipChunk);
	}

	/**
	 * @brief Convert a list of chunks into unique visitors.
	 * Unique visitors are cached into an internal memory.
	 * @param chunk An array of pointers to chunk.
	 * @return An array of unique visitor cache entry.
	 * To obtain the raw unique visitor:
	 * @see ownUniqueChunk()
	*/
	inline STPUniqueChunkCacheEntry cacheUniqueChunk(const STPChunkRecord& chunk) {
		unique_lock<mutex> cache_lock(this->UniqueChunkCacheLock);

		STPUniqueChunkRecord& visitor_entry = this->UniqueChunkCache.emplace_back();
		visitor_entry.reserve(chunk.size());
		
		std::transform(chunk.cbegin(), chunk.cend(), std::back_inserter(visitor_entry), [](auto* chk) {
			//create a unique visitor and insert into cache
			return STPChunk::STPUniqueMapVisitor(*chk);
		});

		return --this->UniqueChunkCache.end();
	}

	/**
	 * @brief Convert a unique chunk visitor to an owning state.
	 * @param entry The iterators to lookup the visitor in the cache entry.
	 * It will remove this visitor from the internal cache such that the iterator will become invalid after return of this function.
	 * @return The unique map visitor.
	*/
	inline STPUniqueChunkRecord ownUniqueChunk(STPUniqueChunkCacheEntry entry) {
		unique_lock<mutex> cache_lock(this->UniqueChunkCacheLock);

		//own the visitor
		STPUniqueChunkRecord rec = std::move(*entry);
		//remove the iterator from the container
		this->UniqueChunkCache.erase(entry);

		return rec;
	}

	/**
	 * @brief Dispatch compute for heightmap, the heightmap result will be written back to the storage
	 * @param current_chunk The maps for the chunk
	 * @param neighbour_chunk The maps of the chunks that require to be used for biome-edge interpolation during heightmap generation,
	 * require the central chunk and neighbour chunks arranged in row-major flavour. The central chunk should also be included.
	 * @param chunkCoord The world coordinate of the chunk
	*/
	void computeHeightmap(STPChunk::STPUniqueMapVisitor& current_chunk, STPUniqueChunkRecord& neighbour_chunk, const ivec2& chunkCoord) {
		//generate heightmap
		STPHeightfieldGenerator::STPMapStorage maps;
		maps.Biomemap.reserve(neighbour_chunk.size());
		maps.Heightmap32F.reserve(1ull);
		for (auto& chk : neighbour_chunk) {
			maps.Biomemap.push_back(chk.biomemap());
		}
		maps.Heightmap32F.push_back(current_chunk.heightmap());
		maps.HeightmapOffset = static_cast<vec2>(this->calcOffset(chunkCoord));
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
	void computeErosion(STPUniqueChunkRecord& neighbour_chunk) {
		STPHeightfieldGenerator::STPMapStorage maps;
		maps.Heightmap32F.reserve(neighbour_chunk.size());
		maps.Heightfield16UI.reserve(neighbour_chunk.size());
		for (auto& chk : neighbour_chunk) {
			maps.Heightmap32F.push_back(chk.heightmap());
			maps.Heightfield16UI.push_back(chk.heightmapBuffer());
		}
		const STPHeightfieldGenerator::STPGeneratorOperation op =
			STPHeightfieldGenerator::Erosion |
			STPHeightfieldGenerator::RenderingBufferGeneration;

		//computing and return success state
		this->generateHeightfield(maps, op);
	}

	/**
	 * @brief Recursively prepare neighbour chunks for the central chunk.
	 * The first recursion will prepare neighbour biomemap for heightmap generation.
	 * The second recursion will prepare neighbour heightmap for erosion.
	 * @param chunkCoord The coordinate to the chunk which should be prepared.
	 * @param rec_depth Please leave this empty, this is the recursion depth and will be managed properly
	 * @return If all neighbours are ready to be used, true is returned.
	 * If any neighbour is not ready (being used by other threads or neighbour is not ready and compute is launched), return false
	*/
	const optional<STPChunk::STPSharedMapVisitor> recNeighbourChecking(const ivec2& chunkCoord, unsigned char rec_depth = 2u) {
		//recursive case:
		//define what rec_depth means...
		constexpr static unsigned char BIOMEMAP_PASS = 1u,
			HEIGHTMAP_PASS = 2u;

		using std::move;
		{
			STPChunk::STPChunkState expected_state = STPChunk::STPChunkState::Empty;
			switch (rec_depth) {
			case BIOMEMAP_PASS: expected_state = STPChunk::STPChunkState::HeightmapReady;
				break;
			case HEIGHTMAP_PASS: expected_state = STPChunk::STPChunkState::Complete;
				break;
			default:
				break;
			}
			if (auto center = this->ChunkCache.find(chunkCoord);
				center != this->ChunkCache.end() && center->second.chunkState() >= expected_state) {
				if (center->second.occupied()) {
					//central chunk is in-used, do not proceed.
					return nullopt;
				}
				//no need to continue if centre chunk is available
				//since the centre chunk might be used as a neighbour chunk later, we only return boolean instead of a pointer
				//after checkChunk() is performed for every chunks, we can grab all pointers and check for occupancy in other functions.
				return make_optional<STPChunk::STPSharedMapVisitor>(center->second);
			}
		}
		auto biomemap_computer = [this](STPUniqueChunkCacheEntry chunk_entry, dvec2 offset) -> void {
			//own the unique chunk visitor
			STPChunk::STPUniqueMapVisitor chunk = move(this->ownUniqueChunk({ chunk_entry }).front());

			//since biomemap is discrete, we need to round the pixel
			ivec2 rounded_offset = static_cast<ivec2>(glm::round(offset));
			STORE_EXCEPTION(this->generateBiomemap(chunk.biomemap(), ivec3(rounded_offset.x, 0, rounded_offset.y)))
			
			//computation was successful
			chunk->markChunkState(STPChunk::STPChunkState::BiomemapReady);
			//chunk will be unlocked automatically by unique visitor
		};
		auto heightmap_computer = [this](STPUniqueChunkCacheEntry neighbours_entry, ivec2 coordinate) -> void {
			STPUniqueChunkRecord neighbours = this->ownUniqueChunk(neighbours_entry);
			//the centre chunk is always at the middle of all neighbours
			STPChunk::STPUniqueMapVisitor& centre = neighbours[neighbours.size() / 2u];

			STORE_EXCEPTION(this->computeHeightmap(centre, neighbours, coordinate))
				
			//computation was successful
			centre->markChunkState(STPChunk::STPChunkState::HeightmapReady);
		};
		auto erosion_computer = [this](STPUniqueChunkCacheEntry neighbours_entry) -> void {
			STPUniqueChunkRecord neighbours = this->ownUniqueChunk(neighbours_entry);

			STORE_EXCEPTION(this->computeErosion(neighbours))
				
			//erosion was successful
			//mark centre chunk complete
			neighbours[neighbours.size() / 2u]->markChunkState(STPChunk::STPChunkState::Complete);
		};

		//reminder: central chunk is included in neighbours
		const STPEnvironment::STPChunkSetting& chk_config = this->ChunkSetting;
		const STPChunk::STPChunkCoordinateCache neighbour_position = this->getNeighbour(chunkCoord);

		bool canContinue = true;
		//The first pass: check if all neighbours are ready for some operations
		STPChunkRecord neighbour;
		for (const auto& neighbourPos : neighbour_position) {
			//get current neighbour chunk
			const auto [chunk_it, chunk_added] = this->ChunkCache.try_emplace(neighbourPos, chk_config.MapSize);
			STPChunk& curr_neighbour = chunk_it->second;

			if (curr_neighbour.occupied()) {
				//occupied means it's currently in used (probably another thread has already started to compute it)
				canContinue = false;
				continue;
			}
			switch (rec_depth) {
			case BIOMEMAP_PASS:
				//container will guaranteed to exists since heightmap pass has already created it
				if (curr_neighbour.chunkState() == STPChunk::STPChunkState::Empty) {
					//compute biomemap
					this->GeneratorWorker.enqueue_void(biomemap_computer, this->cacheUniqueChunk({ &curr_neighbour }), this->calcOffset(neighbourPos));
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

			neighbour.emplace_back(&curr_neighbour);
			//if chunk is found, we can guarantee it's in-used empty or at least biomemap/heightmap complete
		}
		if (!canContinue) {
			//if biomemap/heightmap is computing, we don't need to check for heightmap generation/erosion because some chunks are in use
			return nullopt;
		}

		//The second pass: launch compute on the centre with all neighbours
		//all chunks are available, obtain unique visitors
		const STPUniqueChunkCacheEntry visitor_entry = this->cacheUniqueChunk(neighbour);
		//send the list of neighbour chunks to GPU to perform some operations
		switch (rec_depth) {
		case BIOMEMAP_PASS:
			//generate heightmap
			this->GeneratorWorker.enqueue_void(heightmap_computer, visitor_entry, chunkCoord);
			break;
		case HEIGHTMAP_PASS:
			//perform erosion on heightmap
			this->GeneratorWorker.enqueue_void(erosion_computer, visitor_entry);
			{
				//trigger a chunk reload as some chunks have been added to render buffer already after neighbours are updated
				for_each(neighbour_position.cbegin(), neighbour_position.cend(), 
					[&pipeline = this->Pipeline](const auto& position) -> void { pipeline.reload(position); });
			}
			break;
		default:
			//never gonna happen
			break;
		}

		//compute has been launched
		return nullopt;
	}

public:

	/**
	 * @brief Initialise generator manager with pipeline stages filled with generators.
	 * @param setup The pointer to all pipeline stages.
	 * @param pipeline The pointer to the world pipeline registered with the generator manager.
	*/
	STPGeneratorManager(STPWorldPipeline::STPPipelineSetup& setup, STPWorldPipeline& pipeline) : 
		generateBiomemap(*setup.BiomemapGenerator), generateHeightfield(*setup.HeightfieldGenerator), generateSplatmap(*setup.SplatmapGenerator), 
		Pipeline(pipeline), ChunkSetting(this->Pipeline.ChunkSetting), GeneratorWorker(5u) {

	}

	STPGeneratorManager(const STPGeneratorManager&) = delete;

	STPGeneratorManager(STPGeneratorManager&&) = delete;

	STPGeneratorManager& operator=(const STPGeneratorManager&) = delete;

	STPGeneratorManager& operator=(STPGeneratorManager&&) = delete;

	~STPGeneratorManager() = default;

	/**
	 * @brief Request a pointer to the chunk given a world coordinate.
	 * @param chunk_coord The world coordinate where the chunk is requesting.
	 * @return The shared visitor to the requested chunk.
	 * The function returns a valid point only when the chunk is fully ready for rendering.
	 * In case chunk is not ready, such as being used by other chunks, or map generation is in progress, nullptr is returned.
	*/
	auto getChunk(const ivec2& chunk_coord) {
		//check if there's any exception thrown from previous asynchronous compute launch
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
		return this->recNeighbourChecking(chunk_coord);
	}

	/**
	 * @brief Prepare and generate splatmap for rendering.
	 * @tparam M The terrain map memory.
	 * @param buffer The terrain map on device side, a mapped OpenGL pointer.
	 * @param requesting_chunk Specify chunks that need to have splatmap generated.
	 * Note that the coordinate of local chunk should specify chunk map offset rather than chunk world position.
	*/
	template<class M>
	void computeSplatmap(const M& buffer,
		const STPTextureFactory::STPRequestingChunkInfo& requesting_chunk) const {
		//prepare texture and surface object
		cudaTextureObject_t biomemap, heightfield;
		cudaSurfaceObject_t splatmap;
		//make sure all description are zero init
		cudaResourceDesc res_desc = {};
		res_desc.resType = cudaResourceTypeArray;
		cudaTextureDesc tex_desc = {};
		tex_desc.addressMode[0] = cudaAddressModeClamp;
		tex_desc.addressMode[1] = cudaAddressModeClamp;
		tex_desc.addressMode[2] = cudaAddressModeClamp;
		tex_desc.normalizedCoords = 0;

		//biomemap
		res_desc.res.array.array = buffer[TERRAIN_MAP_INDEX(STPWorldPipeline::STPTerrainMapType::Biomemap)];
		tex_desc.filterMode = cudaFilterModePoint;
		tex_desc.readMode = cudaReadModeElementType;
		STPcudaCheckErr(cudaCreateTextureObject(&biomemap, &res_desc, &tex_desc, nullptr));

		//heightfield
		res_desc.res.array.array = buffer[TERRAIN_MAP_INDEX(STPWorldPipeline::STPTerrainMapType::Heightmap)];
		tex_desc.filterMode = cudaFilterModeLinear;
		tex_desc.readMode = cudaReadModeNormalizedFloat;
		STPcudaCheckErr(cudaCreateTextureObject(&heightfield, &res_desc, &tex_desc, nullptr));

		//splatmap
		res_desc.res.array.array = buffer[TERRAIN_MAP_INDEX(STPWorldPipeline::STPTerrainMapType::Splatmap)];
		STPcudaCheckErr(cudaCreateSurfaceObject(&splatmap, &res_desc));

		const cudaStream_t stream = *this->Pipeline.BufferStream;
		//launch splatmap computer
		this->generateSplatmap(biomemap, heightfield, splatmap, requesting_chunk, stream);

		//before deallocation happens make sure everything has finished.
		STPcudaCheckErr(cudaStreamSynchronize(stream));
		//finish up, we have to delete it every time because resource array may change every time we re-map it
		STPcudaCheckErr(cudaDestroyTextureObject(biomemap));
		STPcudaCheckErr(cudaDestroyTextureObject(heightfield));
		STPcudaCheckErr(cudaDestroySurfaceObject(splatmap));
	}

};

#define FOR_EACH_MEMORY_START() for (size_t i = 0u; i < STPWorldPipeline::STPMemoryManager::MemoryUnitCapacity; i++) {
#define FOR_EACH_MEMORY_END() }

class STPWorldPipeline::STPMemoryManager {
public:

	//The number of map data in a memory unit.
	constexpr static size_t MemoryUnitCapacity = 3ull;
	//Channel size in byte (not bit) for each map.
	constexpr static size_t MemoryFormat[3] = {
		sizeof(Sample),
		sizeof(unsigned short),
		sizeof(unsigned char)
	};
	//Maximum value of memory format
	constexpr static size_t MaxMemoryFormat = *std::max_element(MemoryFormat, MemoryFormat + MemoryUnitCapacity);

	/**
	 * @brief STPMemoryBlock is a collection of memory units containing a complete set of world data.
	*/
	struct STPMemoryBlock {
	public:

		//A memory unit in a memory block is a single piece of memory of one type of world data.
		template<typename T>
		using STPMemoryUnit = array<T, MemoryUnitCapacity>;
		//Terrain maps that are mapped to CUDA array.
		using STPMappedMemoryUnit = STPMemoryUnit<cudaArray_t>;

		//index 0: R16UI biome map
		//index 1: R16 height map
		//index 2: R8UI splat map
		STPMemoryUnit<GLuint> TerrainMap;
		//registered CUDA graphics handle for GL terrain maps.
		STPMemoryUnit<cudaGraphicsResource_t> TerrainMapResource;

		//A status flag indicating if all local chunks in this memory block
		//have their memory loaded from the host memory.
		atomic_bool LocalChunkComplete;
		//Vector that stored rendered chunk world position and loading status (True is loaded, false otherwise).
		vector<pair<ivec2, bool>> LocalChunkRecord;
		//Use chunk world coordinate to lookup chunk ID, which is the index to the local chunk record.
		unordered_map<ivec2, unsigned int, STPHashivec2> LocalChunkDictionary;

		/**
		 * @brief Initialise a memory block.
		 * @param manager The pointer to the dependent world memory manager.
		*/
		STPMemoryBlock(const STPMemoryManager& manager) : LocalChunkComplete(true) {
			const STPEnvironment::STPChunkSetting& setting = manager.ChunkSetting;
			const uvec2 buffer_size = setting.RenderedChunk * setting.MapSize;
			auto setupMap = 
				[buffer_size](GLuint texture, GLint min_filter, GLint mag_filter, GLsizei levels, GLenum internal) -> void {
				//set texture parameter
				glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
				glTextureParameteri(texture, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
				glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, min_filter);
				glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, mag_filter);
				//allocation
				glTextureStorage2D(texture, levels, internal, buffer_size.x, buffer_size.y);
			};
			auto registerMap = 
				[](GLuint texture, cudaGraphicsResource_t* res, unsigned int reg_flag = cudaGraphicsRegisterFlagsNone) -> void {
				//register CUDA texture
				STPcudaCheckErr(cudaGraphicsGLRegisterImage(res, texture, GL_TEXTURE_2D, reg_flag));
				STPcudaCheckErr(cudaGraphicsResourceSetMapFlags(*res, cudaGraphicsMapFlagsNone));
			};

			//creating texture map
			glCreateTextures(GL_TEXTURE_2D, 3, this->TerrainMap.data());
			//biomemap
			setupMap(this->TerrainMap[0], GL_NEAREST, GL_NEAREST, 1, GL_R16UI);
			registerMap(this->TerrainMap[0], &this->TerrainMapResource[0]);
			//heightfield
			setupMap(this->TerrainMap[1], GL_LINEAR, GL_LINEAR, 1, GL_R16);
			registerMap(this->TerrainMap[1], &this->TerrainMapResource[1]);
			//splatmap
			setupMap(this->TerrainMap[2], GL_NEAREST, GL_NEAREST, 1, GL_R8UI);
			registerMap(this->TerrainMap[2], &this->TerrainMapResource[2], cudaGraphicsRegisterFlagsSurfaceLoadStore);

			const unsigned int renderedChunkCount = setting.RenderedChunk.x * setting.RenderedChunk.y;
			//reserve memory for lookup tables
			this->LocalChunkRecord.reserve(renderedChunkCount);
			this->LocalChunkDictionary.reserve(renderedChunkCount);
		}

		STPMemoryBlock(const STPMemoryBlock&) = delete;

		STPMemoryBlock(STPMemoryBlock&&) = delete;

		STPMemoryBlock& operator=(const STPMemoryBlock&) = delete;

		STPMemoryBlock& operator=(STPMemoryBlock&&) = delete;

		~STPMemoryBlock() {
			//unregister texture
			FOR_EACH_MEMORY_START()
				STPcudaCheckErr(cudaGraphicsUnregisterResource(this->TerrainMapResource[i]));
			FOR_EACH_MEMORY_END()

			//delete texture map
			glDeleteTextures(3, this->TerrainMap.data());
		}

		/**
		 * @brief Set the CUDA resource mapping flag for each GL texture.
		 * @param flag The resource mapping flag.
		*/
		inline void setMappingFlag(unsigned int flag) const {
			FOR_EACH_MEMORY_START()
				STPcudaCheckErr(cudaGraphicsResourceSetMapFlags(this->TerrainMapResource[i], flag));
			FOR_EACH_MEMORY_END()
		}

		/**
		 * @brief Copy the local chunk lookup table from one memory block to the current one.
		 * @param block The memory block from which is copied in.
		*/
		inline void copyLocalChunkTable(const STPMemoryBlock& block) {
			//copy local status record table
			this->LocalChunkRecord.clear();
			std::copy(block.LocalChunkRecord.cbegin(), block.LocalChunkRecord.cend(),
				std::back_inserter(this->LocalChunkRecord));
			this->LocalChunkComplete = static_cast<bool>(block.LocalChunkComplete);

			//copy local index lookup table
			this->LocalChunkDictionary.clear();
			for_each(block.LocalChunkDictionary.cbegin(), block.LocalChunkDictionary.cend(),
				[chunkIdx = 0u, &local_dict = this->LocalChunkDictionary](const auto local) mutable {
				local_dict.try_emplace(local.first, chunkIdx++);
			});
		}

		/**
		 * @brief Recompute the chunk tables.
		 * @param rendered_chunk A range of rendered chunks.
		*/
		inline void recomputeLocalChunkTable(const STPChunk::STPChunkCoordinateCache& rendered_chunk) {
			this->LocalChunkRecord.clear();
			this->LocalChunkDictionary.clear();
			this->LocalChunkComplete = false;
			//we also need chunkID, which is just the index of the visible chunk from top-left to bottom-right
			for_each(rendered_chunk.cbegin(), rendered_chunk.cend(),
				[chunkIdx = 0u, &local_status = this->LocalChunkRecord,
					&local_dict = this->LocalChunkDictionary](const auto chunk_pos) mutable {
				local_status.emplace_back(chunk_pos, false);
				local_dict.try_emplace(chunk_pos, chunkIdx++);
			});
		}

		/**
		 * @brief Map the GL terrain texture to CUDA array.
		 * @param stream The CUDA stream where the work is submitted to.
		 * @return The mapped CUDA array.
		*/
		inline STPMappedMemoryUnit mapTerrainMap(cudaStream_t stream) {
			STPMappedMemoryUnit map;
			//map the texture, all OpenGL related work must be done on the main contexted thread
			//CUDA will make sure all previous graphics API calls are finished before stream begins
			STPcudaCheckErr(cudaGraphicsMapResources(3, this->TerrainMapResource.data(), stream));
			//we only have one texture, so index is always zero
			FOR_EACH_MEMORY_START()
				STPcudaCheckErr(cudaGraphicsSubResourceGetMappedArray(&map[i], this->TerrainMapResource[i], 0, 0));
			FOR_EACH_MEMORY_END()

			return map;
		}

		/**
		 * @brief Unmap the GL terrain texture from CUDA array.
		 * @param stream The CUDA stream where the unmap should be synchronised.
		*/
		inline void unmapTerrainmap(cudaStream_t stream) {
			//sync the stream that modifies the texture and unmap the chunk
			//CUDA will make sure all previous pending works in the stream has finished before graphics API can be called
			STPcudaCheckErr(cudaGraphicsUnmapResources(3, this->TerrainMapResource.data(), stream));
		}

		/**
		 * @brief Force reload a terrain map.
		 * @param chunkPos The chunk position to be reloaded.
		 * @return The status indicating if the chunk position is in the local rendered area,
		 * and if the reload flag has been set correctly.
		*/
		inline bool reloadMap(const ivec2& chunkPos) {
			auto it = this->LocalChunkDictionary.find(chunkPos);
			if (it == this->LocalChunkDictionary.end()) {
				//chunk position provided is not required to be rendered, or new rendering area has changed
				return false;
			}
			//found, trigger a reload
			this->LocalChunkRecord[it->second].second = false;
			this->LocalChunkComplete = false;
			return true;
		}

	};

private:

	STPWorldPipeline& Pipeline;
	const STPEnvironment::STPChunkSetting& ChunkSetting;

	STPMemoryBlock MemoryBuffer[2];
	//The transfer cache is used because cudaMemcpy2DArrayToArray does not have a streamed version
	STPSmartDeviceMemory::STPPitchedDeviceMemory<unsigned char[]> MapClearBuffer, MapTransferCache;

	//Record indices of chunks that need to have splatmap computed for the current task.
	unordered_set<unsigned int> ComputingChunk;
	constexpr static array<ivec2, 8ull> NeighbourCoordinateOffset = {
		ivec2(-1, -1),
		ivec2(+0, -1),
		ivec2(+1, -1),
		ivec2(-1, +0),
		ivec2(+1, +0),
		ivec2(-1, +1),
		ivec2(+0, +1),
		ivec2(+1, +1)
	};

	/**
	 * @brief Calculate the map offset on the terrain map to reach a specific local chunk.
	 * @param index The local chunk index.
	 * @return The chunk memory offset on the map.
	*/
	inline uvec2 calcLocalMapOffset(unsigned int index) const noexcept {
		const STPEnvironment::STPChunkSetting& chunk_setting = this->ChunkSetting;
		const unsigned int rendered_width = chunk_setting.RenderedChunk.x;

		//calculate global offset, basically
		const uvec2 chunkIdx(index % rendered_width, index / rendered_width);
		return chunk_setting.MapSize * chunkIdx;
	}

public:

	//Double buffering.
	//Front buffer is passed to the user.
	const STPMemoryBlock* FrontBuffer;
	//Back buffer holds the currently in-progress generation.
	STPMemoryBlock* BackBuffer;

private:

	//Store the CUDA array after mapping the back buffer
	STPMemoryBlock::STPMappedMemoryUnit MappedBackBuffer;
	bool isBackBufferMapped;

	/**
	 * @brief Correct the CUDA graphics mapping flag for the buffers.
	*/
	inline void correctResourceMappingFlag() {
		this->FrontBuffer->setMappingFlag(cudaGraphicsMapFlagsReadOnly);
		this->BackBuffer->setMappingFlag(cudaGraphicsMapFlagsNone);
	}

public:

	/**
	 * @brief Initialise a world memory manager.
	 * @param pipeline The pointer to the dependent world pipeline.
	*/
	STPMemoryManager(STPWorldPipeline& pipeline) :
		Pipeline(pipeline), ChunkSetting(this->Pipeline.ChunkSetting),
		MemoryBuffer { STPMemoryBlock(*this), STPMemoryBlock(*this) },
		FrontBuffer(this->MemoryBuffer), BackBuffer(this->MemoryBuffer + 1u), isBackBufferMapped(false) {
		const STPEnvironment::STPChunkSetting& setting = this->ChunkSetting;
		const uvec2 mapDim = setting.MapSize, 
			clearBuffer_size = mapDim * uvec2(MaxMemoryFormat, 1u);
		const unsigned int chunkCount = setting.RenderedChunk.x * setting.RenderedChunk.y;

		//init clear buffers that are used to clear texture when new rendered chunks are loaded
		//(we need to clear the previous chunk data)
		this->MapClearBuffer =
			STPSmartDeviceMemory::makePitchedDevice<unsigned char[]>(clearBuffer_size.x, clearBuffer_size.y);
		this->MapTransferCache =
			STPSmartDeviceMemory::makePitchedDevice<unsigned char[]>(clearBuffer_size.x, clearBuffer_size.y);

		const cudaStream_t stream = *this->Pipeline.BufferStream;
		//clear the `clear` buffer
		const auto& [clear_ptr, clear_pitch] = this->MapClearBuffer;
		STPcudaCheckErr(
			cudaMemset2DAsync(clear_ptr.get(), clear_pitch, 0x80u, clearBuffer_size.x, clearBuffer_size.y, stream));

		//initialise pixel value in the texture rather than leaving them as undefined.
		for (int b = 0; b < 2; b++) {
			STPMemoryBlock& block = this->MemoryBuffer[b];
			const auto& mapped = block.mapTerrainMap(stream);

			//clear every chunk
			for (unsigned int y = 0u; y < setting.RenderedChunk.y; y++) {
				for (unsigned int x = 0u; x < setting.RenderedChunk.x; x++) {
					const uvec2 localMapOffset = uvec2(x, y) * setting.MapSize;
					FOR_EACH_MEMORY_START()
						STPcudaCheckErr(cudaMemcpy2DToArrayAsync(mapped[i], localMapOffset.x * MemoryFormat[i], localMapOffset.y,
							clear_ptr.get(), clear_pitch, mapDim.x * MemoryFormat[i], mapDim.y,
							cudaMemcpyDeviceToDevice, stream));
					FOR_EACH_MEMORY_END()
				}
			}

			block.unmapTerrainmap(stream);
		}
		//clean up
		STPcudaCheckErr(cudaStreamSynchronize(stream));
		this->correctResourceMappingFlag();

		this->ComputingChunk.reserve(chunkCount);
	}

	STPMemoryManager(const STPMemoryManager&) = delete;

	STPMemoryManager(STPMemoryManager&&) = delete;

	STPMemoryManager& operator=(const STPMemoryManager&) = delete;

	STPMemoryManager& operator=(STPMemoryManager&&) = delete;

	~STPMemoryManager() = default;

	/**
	 * @brief Get the corresponding terrain map given a type from the front buffer.
	 * @param type The type of the map to be retrieved.
	*/
	inline GLuint getMap(STPWorldPipeline::STPTerrainMapType type) const noexcept {
		return this->FrontBuffer->TerrainMap[TERRAIN_MAP_INDEX(type)];
	}

	/**
	 * @brief Map the back buffer.
	 * If the back buffer is already mapped, no operation is performed.
	*/
	inline void mapBackBuffer() {
		if (this->isBackBufferMapped) {
			return;
		}
		this->MappedBackBuffer = this->BackBuffer->mapTerrainMap(*this->Pipeline.BufferStream);
		this->isBackBufferMapped = true;
	}

	/**
	 * @brief Unmap the back buffer.
	 * The mapped back buffer array becomes undefined.
	 * If the back buffer is not mapped, no operation is performed.
	*/
	inline void unmapBackBuffer() {
		if (!this->isBackBufferMapped) {
			return;
		}
		this->BackBuffer->unmapTerrainmap(*this->Pipeline.BufferStream);
		this->isBackBufferMapped = false;
	}

	/**
	 * @brief Get the array to the mapped back buffer.
	 * @return The array to the mapped back buffer.
	*/
	inline const STPMemoryBlock::STPMappedMemoryUnit& getMappedBackBuffer() const {
		return this->MappedBackBuffer;
	}

	/**
	 * @brief Swap the pointer of front and back buffer.
	*/
	inline void swapBuffer() {
		std::swap(const_cast<STPMemoryBlock*&>(this->FrontBuffer), this->BackBuffer);
		this->correctResourceMappingFlag();
	}

	/**
	 * @brief Attempt to reuse as many chunks in the back buffer as possible from the front buffer.
	 * If a chunk cannot be reused, it will be cleared instead.
	 * @param backBuffer_ptr The mapped back buffer.
	*/
	void reuseBuffer(const STPMemoryBlock::STPMappedMemoryUnit& backBuffer_ptr) {
		const STPEnvironment::STPChunkSetting& chunk_setting = this->ChunkSetting;
		const uvec2 mapDim = chunk_setting.MapSize;

		cudaStream_t stream = *this->Pipeline.BufferStream;
		//map the front buffer as read-only
		STPMemoryBlock* frontBuffer_w = const_cast<STPMemoryBlock*>(this->FrontBuffer);
		const auto frontBuffer_ptr = frontBuffer_w->mapTerrainMap(stream);

		bool allLoaded = true;
		const auto& front_local_dict = this->FrontBuffer->LocalChunkDictionary;
		//here we perform an optimisation: reuse chunk that has been rendered previously from the front buffer
		for (auto& [chunkPos, loaded] : this->BackBuffer->LocalChunkRecord) {
			//checking the new back buffer chunk, is there any old chunk has the same world coordinate as the new chunk?
			auto equal_pos_it = front_local_dict.find(chunkPos);

			const unsigned int back_buffer_chunkIdx = this->BackBuffer->LocalChunkDictionary.at(chunkPos);
			if (equal_pos_it != front_local_dict.cend()) {
				//found, check if the previous cache is complete
				const unsigned int front_buffer_chunkIdx = equal_pos_it->second;
				const bool front_status = this->FrontBuffer->LocalChunkRecord[front_buffer_chunkIdx].second;

				if (front_status) {
					//if the previous front buffer chunk is complete, copy to the back buffer
					{
						const uvec2 src_offset = this->calcLocalMapOffset(front_buffer_chunkIdx),
							dest_offset = this->calcLocalMapOffset(back_buffer_chunkIdx);
						const auto& [trans_ptr, trans_pitch] = this->MapTransferCache;

						//no async version for function cudaMemcpy2DArrayToArray, wtf???
						FOR_EACH_MEMORY_START()
							//front buffer -> cache
							STPcudaCheckErr(cudaMemcpy2DFromArrayAsync(trans_ptr.get(), trans_pitch, frontBuffer_ptr[i],
								src_offset.x * MemoryFormat[i], src_offset.y, mapDim.x * MemoryFormat[i], mapDim.y,
								cudaMemcpyDeviceToDevice, stream));
							//cache -> back buffer
							STPcudaCheckErr(cudaMemcpy2DToArrayAsync(backBuffer_ptr[i], dest_offset.x * MemoryFormat[i],
								dest_offset.y, trans_ptr.get(), trans_pitch, mapDim.x * MemoryFormat[i], mapDim.y,
								cudaMemcpyDeviceToDevice, stream));
						FOR_EACH_MEMORY_END()
					}

					//mark this chunk is loaded in the back buffer
					loaded = true;
					continue;
				}
			}
			//the current back buffer chunk has no usable chunk, we need to load it from chunk storage later
			//clear this chunk
			{
				const uvec2 dest_offset = this->calcLocalMapOffset(back_buffer_chunkIdx);
				const auto& [clear_ptr, clear_pitch] = this->MapClearBuffer;
				FOR_EACH_MEMORY_START()
					STPcudaCheckErr(cudaMemcpy2DToArrayAsync(backBuffer_ptr[i], dest_offset.x * MemoryFormat[i],
						dest_offset.y, clear_ptr.get(), clear_pitch, mapDim.x * MemoryFormat[i], mapDim.y,
						cudaMemcpyDeviceToDevice, stream));
				FOR_EACH_MEMORY_END()
			}
			//there is at least one chunk not loaded, so the entire terrain map is not complete
			allLoaded = false;
		}
		//update overall chunk complete status for back buffer
		this->BackBuffer->LocalChunkComplete = allLoaded;

		//unmap the front buffer
		frontBuffer_w->unmapTerrainmap(stream);

		//record indices of chunks that need to be computed
		const auto& backBuf_rec = this->BackBuffer->LocalChunkRecord;
		this->ComputingChunk.clear();
		//To avoid artefacts, if any chunk is a neighbour of those chunks we just recorded, need to recompute them as
		//well. This is to mainly avoid splatmap seams. The logic is, if previously those chunks do not have a valid
		//neighbour but this time they have one, The border of them may not be aligned properly with the new chunks.
		const uvec2 rendered_chunk = chunk_setting.RenderedChunk;
		const unsigned int maxChunkIdx = rendered_chunk.x * rendered_chunk.y,
			chunkRowCount = chunk_setting.RenderedChunk.x;
		for (unsigned int chunkIdx = 0u; chunkIdx < backBuf_rec.size(); chunkIdx++) {
			if (backBuf_rec[chunkIdx].second) {
				//if this chunk already has data, skip it along with its neighbours.
				continue;
			}

			//for each neighbour of each computing chunk...
			for (const ivec2 coord_offset : this->NeighbourCoordinateOffset) {
				//small negative number will become a huge unsigned,
				//so when checking for validity we don't need to consider negative number.
				const uvec2 current_coord =
					static_cast<uvec2>(ivec2(chunkIdx % chunkRowCount, chunkIdx / chunkRowCount) + coord_offset);
				//make sure the coordinate is in a valid range
				if (current_coord.x >= rendered_chunk.x || current_coord.y >= rendered_chunk.y) {
					continue;
				}

				this->ComputingChunk.emplace(current_coord.x + current_coord.y * chunkRowCount);
			}
		}
	}

	/**
	 * @brief Trigger a chunk buffer reload for the back buffer.
	 * @param chunkCoord The chunk coordinate for the chunk to be reloaded.
	 * @return True if the chunk coordinate is triggered to be reloaded.
	 * False if chunk coordinate is not in rendered area.
	*/
	inline bool reloadBuffer(ivec2 chunkCoord) {
		const bool reload_status = this->BackBuffer->reloadMap(chunkCoord);
		if (reload_status) {
			//also trigger a re-compute if it has been reloaded
			this->ComputingChunk.emplace(this->BackBuffer->LocalChunkDictionary.at(chunkCoord));
		}

		return reload_status;
	}

	/**
	 * @brief Transfer terrain map on host side to device (OpenGL) texture by a local chunk.
	 * @param buffer Texture map on device side, a mapped OpenGL pointer.
	 * @param chunk_visitor The visitor to read the chunk data.
	 * @param chunkIdx Local chunk index that specified which chunk in render area will be overwritten.
	*/
	void sendChunkToBuffer(const STPMemoryBlock::STPMappedMemoryUnit& buffer,
		const STPChunk::STPSharedMapVisitor& chunk_visitor, unsigned int chunkIdx) {
		//chunk is ready, copy to rendering buffer
		auto copy_buffer = [stream = *this->Pipeline.BufferStream, dimension = this->ChunkSetting.MapSize,
			buffer_offset = this->calcLocalMapOffset(chunkIdx)]
			(cudaArray_t dest, const void* src, size_t channelSize) -> void {
			STPcudaCheckErr(cudaMemcpy2DToArrayAsync(dest, buffer_offset.x * channelSize, buffer_offset.y, src,
				dimension.x * channelSize, dimension.x * channelSize, dimension.y, cudaMemcpyHostToDevice, stream));
		};

		unsigned int index;
		//copy buffer to GL texture
		index = TERRAIN_MAP_INDEX(STPWorldPipeline::STPTerrainMapType::Biomemap);
		copy_buffer(buffer[index], chunk_visitor.biomemap(), MemoryFormat[index]);
		index = TERRAIN_MAP_INDEX(STPWorldPipeline::STPTerrainMapType::Heightmap);
		copy_buffer(buffer[index], chunk_visitor.heightmapBuffer(), MemoryFormat[index]);
	}

	/**
	 * @brief Copy the local chunk lookup table from the front buffer to the back buffer.
	*/
	inline void copyLocalChunkTable() {
		this->BackBuffer->copyLocalChunkTable(*this->FrontBuffer);
	}

	/**
	 * @brief Recompute the local chunk status record and index lookup table.
	 * @param chunkPos The new chunk position.
	*/
	inline void recomputeLocalChunkTable(const ivec2& chunkPos) {
		const STPEnvironment::STPChunkSetting& setting = this->ChunkSetting;

		STPChunk::STPChunkCoordinateCache allChunks =
			STPChunk::calcChunkNeighbour(chunkPos, setting.ChunkSize, setting.RenderedChunk);
		this->BackBuffer->recomputeLocalChunkTable(allChunks);
	}

	/**
	 * @brief Generate a list of chunks that are required to have splatmap computed, based on the compute table.
	 * Therefore splatmap of all chunks should be computed after all chunks have finished.
	 * @return The splatmap generator requesting info.
	*/
	STPDiversity::STPTextureFactory::STPRequestingChunkInfo generateSplatmapGeneratorInfo() const {
		STPDiversity::STPTextureFactory::STPRequestingChunkInfo requesting_info;
		requesting_info.reserve(this->ComputingChunk.size());

		for_each(this->ComputingChunk.cbegin(), this->ComputingChunk.cend(),
			[&requesting_info, &chunk_setting = this->ChunkSetting,
				&chunk_record = std::as_const(this->BackBuffer->LocalChunkRecord)](const unsigned int index) {
				//mark updated rendering buffer
				//we need to use the chunk normalised coordinate to get the splatmap offset,
				//splatmap offset needs to be consistent with the heightmap and biomemap
				const vec2 offset = static_cast<vec2>(STPChunk::calcChunkMapOffset(chunk_record[index].first,
					chunk_setting.ChunkSize, chunk_setting.MapSize, chunk_setting.MapOffset));
				//local chunk coordinate
				const uvec2 local_coord = STPChunk::calcLocalChunkCoordinate(index, chunk_setting.RenderedChunk);
				requesting_info.emplace_back(
					STPDiversity::STPTextureInformation::STPSplatGeneratorInformation::STPLocalChunkInformation
					{ local_coord.x, local_coord.y, offset.x, offset.y }
				);
			});

		return requesting_info;
	}

};

STPWorldPipeline::STPWorldPipeline(STPPipelineSetup& setup) :
	ChunkSetting(*setup.ChunkSetting), BufferStream(cudaStreamNonBlocking),
	Generator(make_unique<STPGeneratorManager>(setup, *this)), Memory(make_unique<STPMemoryManager>(*this)),
	LastCentreLocation(ivec2(std::numeric_limits<int>::min())), PipelineWorker(1u) {
	
}

STPWorldPipeline::~STPWorldPipeline() {
	//sync the chunk loading and make sure the chunk loader has finished before return
	if (this->MapLoader.valid()) {
		//wait for finish first
		this->MapLoader.get();
	}
	this->Memory->unmapBackBuffer();

	//wait for the stream to finish everything
	STPcudaCheckErr(cudaStreamSynchronize(*this->BufferStream));
}

inline STPWorldPipeline::STPChunkLoaderStatus STPWorldPipeline::isLoaderBusy() {
	if (!this->MapLoader.valid()) {
		//no work is being done
		return STPChunkLoaderStatus::Free;
	}
	if (this->MapLoader.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
		//still working
		return STPChunkLoaderStatus::Busy;
	}
	this->MapLoader.get();
	return STPChunkLoaderStatus::Yield;
}

const ivec2& STPWorldPipeline::centre() const noexcept {
	return this->LastCentreLocation;
}

STPWorldPipeline::STPWorldLoadStatus STPWorldPipeline::load(const dvec3& viewPos) {
	const STPEnvironment::STPChunkSetting& chunk_setting = this->ChunkSetting;
	
	/* -------------------------------- Status Check --------------------------------- */
	const STPChunkLoaderStatus loaderStatus = this->isLoaderBusy();
	const bool backBufferReady = this->Memory->BackBuffer->LocalChunkComplete;
	bool shouldClearBuffer = false;

	//check if the central position has changed or not
	if (const ivec2 thisCentrePos = STPChunk::calcWorldChunkCoordinate(
			viewPos - chunk_setting.ChunkOffset, chunk_setting.ChunkSize, chunk_setting.ChunkScaling);
		thisCentrePos != this->LastCentreLocation) {
		//centre position changed
		if (loaderStatus == STPChunkLoaderStatus::Busy) {
			//loader is working on the back buffer for a previous task
			return STPWorldLoadStatus::BufferExhaust;
		}
		//discard previous unfinished task and move on to the new task
		
		//backup the current rendering locals
		this->Memory->copyLocalChunkTable();
		//recalculate loading chunks
		this->Memory->recomputeLocalChunkTable(thisCentrePos);

		this->LastCentreLocation = thisCentrePos;
		//clear up previous rendering buffer
		shouldClearBuffer = true;
		//start loading the new rendered chunks
		this->Memory->mapBackBuffer();
	} else {
		//centre position has not changed
		if (loaderStatus == STPChunkLoaderStatus::Busy) {
			//loader is working on the back buffer for the current task
			return STPWorldLoadStatus::BackBufferBusy;
		}
		if (backBufferReady) {
			if (loaderStatus == STPChunkLoaderStatus::Yield) {
				//current task just done completely
				//synchronise the buffer before passing to the user
				this->Memory->unmapBackBuffer();
				this->Memory->swapBuffer();
				return STPWorldLoadStatus::Swapped;
			}
			return STPWorldLoadStatus::Unchanged;
		}
		//continue loading the current rendered chunks
	}

	/* ----------------------------- Asynchronous Chunk Loading ------------------------------- */
	auto asyncChunkLoader = [&mem_mgr = *this->Memory, &gen_mgr = *this->Generator, &chunk_setting](const auto& map_data) -> void {
		auto& local_record = mem_mgr.BackBuffer->LocalChunkRecord;
		bool allLoaded = true;
		//check all workers and release chunks once they have finished.
		for (unsigned int i = 0u; i < local_record.size(); i++) {
			auto& [chunkPos, chunkLoaded] = local_record[i];
			if (chunkLoaded) {
				//skip this chunk if loading has been completed before
				continue;
			}

			//ask provider if we can get the chunk
			const optional<STPChunk::STPSharedMapVisitor> chunk = gen_mgr.getChunk(chunkPos);
			if (chunk) {
				//load chunk into device texture
				mem_mgr.sendChunkToBuffer(map_data, chunk.value(), i);
				chunkLoaded = true;
				continue;
			}

			//chunk is not gettable
			allLoaded = false;
		}
		//generate splatmap after all chunks had their maps loaded from host, so we only need to generate once.
		if (allLoaded) {
			//there exists chunk that has terrain maps updated, we need to update splatmap as well
			gen_mgr.computeSplatmap(map_data, mem_mgr.generateSplatmapGeneratorInfo());
		}

		//update chunk complete status
		mem_mgr.BackBuffer->LocalChunkComplete = allLoaded;
	};

	/* ----------------------------- Front Buffer Backup -------------------------------- */
	const auto& backBuffer_ptr = this->Memory->getMappedBackBuffer();
	//Search if any chunk in the front buffer can be reused.
	//If so, copy to the back buffer; if not, clear the chunk.
	if (shouldClearBuffer) {
		this->Memory->reuseBuffer(backBuffer_ptr);
	}

	//group mapped data together and start loading chunk
	this->MapLoader = this->PipelineWorker.enqueue_future(asyncChunkLoader, std::cref(backBuffer_ptr));
	return STPWorldLoadStatus::BackBufferBusy;
}

bool STPWorldPipeline::reload(const ivec2& chunkCoord) {
	return this->Memory->reloadBuffer(chunkCoord);
}

STPOpenGL::STPuint STPWorldPipeline::operator[](STPTerrainMapType type) const noexcept {
	return this->Memory->getMap(type);
}

const STPDiversity::STPTextureFactory& STPWorldPipeline::splatmapGenerator() const noexcept {
	return this->Generator->generateSplatmap;
}