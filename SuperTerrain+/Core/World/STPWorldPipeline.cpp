#include <SuperTerrain+/World/STPWorldPipeline.h>

//Data Structure
#include <SuperTerrain+/World/Chunk/STPChunk.h>
#include <SuperTerrain+/Utility/Memory/STPObjectPool.h>

//Error Handling
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

//Hasher
#include <SuperTerrain+/Utility/STPHashCombine.h>

//GL
#include <glad/glad.h>

#include <glm/common.hpp>

//Container
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <queue>
//Thread
#include <future>

#include <utility>
#include <iterator>
#include <limits>
#include <algorithm>
#include <type_traits>
#include <cassert>

//GLM
using glm::ivec2;
using glm::uvec2;
using glm::vec2;
using glm::dvec2;
using glm::dvec3;

using std::array;
using std::pair;
using std::unordered_map;
using std::unordered_multimap;
using std::unordered_set;
using std::vector;
using std::queue;
using std::priority_queue;

using std::unique_ptr;
using std::future;
using std::numeric_limits;

using std::for_each;
using std::transform;
using std::make_unique;
using std::make_pair;
using std::as_const;
using std::back_inserter;

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;

#define TERRAIN_MAP_INDEX(M) static_cast<std::underlying_type_t<STPWorldPipeline::STPTerrainMapType>>(M)

namespace {
	/**
	 * @brief The hash function for the glm::ivec2
	 */
	struct STPHashivec2 {
	public:

		inline size_t operator()(const ivec2& position) const noexcept {
			//combine hash
			return STPHashCombine::combine(0u, position.x, position.y);
		}

	};
}

class STPWorldPipeline::STPGeneratorManager {
private:

	//create a new array of pointers containing exact number of neighbour entry
	struct STPNeighbourArrayCreator {
	public:

		const size_t NeighbourCount;

		/**
		 * @brief Create a new neighbour array.
		 * @param neighbour The number of neighbour.
		*/
		STPNeighbourArrayCreator(const uvec2 neighbour) noexcept : NeighbourCount(neighbour.x * neighbour.y) {

		}

		~STPNeighbourArrayCreator() = default;

		inline unique_ptr<void*[]> operator()() const {
			return make_unique<void*[]>(this->NeighbourCount);
		}

	};

	STPWorldPipeline& Pipeline;
	const STPEnvironment::STPChunkSetting& ChunkSetting;

	typedef unordered_map<ivec2, STPChunk, STPHashivec2> STPChunkDatabase;
	typedef STPChunkDatabase::value_type STPChunkDatabaseEntry;
	STPChunkDatabase ChunkDatabase;

	/* ----------------------------------------- custom data structure --------------------------------------------------- */
	//the schedule priority for a chunk, higher priority gets smaller schedule number.
	typedef unsigned short STPChunkPriorityNumber;
	//0 is a reserved number for a dummy variable, does not represent a valid priority.
	//priority starts from the minimum number, which stands for the highest priority
	constexpr static STPChunkPriorityNumber DummyChunkPriorityNumber = 0u,
		MinimumChunkPriorityNumber = 1u;

	//pair of priority number and chunk world coordinate
	typedef pair<STPChunkPriorityNumber, STPChunkDatabaseEntry*> STPChunkSchedulePriority;
	//compares the schedule priority comparator.
	struct STPChunkScheduleComparator {
	public:

		inline bool operator()(const STPChunkSchedulePriority& left, const STPChunkSchedulePriority& right) const noexcept {
			//smallest number (high priority) first
			return left.first > right.first;
		}

	};
	typedef vector<STPChunkDatabaseEntry*> STPChunkDatabaseEntryCollection;
	//a priority queue indicating parallel computing order of chunks to ensure no race condition when chunk neighbour overlaps
	typedef priority_queue<STPChunkSchedulePriority, vector<STPChunkSchedulePriority>, STPChunkScheduleComparator> STPChunkSchedule;

	/* ------------------------------------------------- memory pool ---------------------------------------------------- */
	const STPChunk::STPChunkNeighbourOffset DiversityNeightbourOffset, ErosionNeighbourOffset;
	STPObjectPool<unique_ptr<void*[]>, STPNeighbourArrayCreator> DiversityNeighbourMemoryPool, ErosionNeighbourMemoryPool;

	//used by function to merge neighbour chunks
	struct STPNeighbourUnionCache {
	public:

		//chunks merged
		STPChunkDatabaseEntryCollection UnionChunk;
		//just a lookup table to make sure there is no duplicate entry in the array above
		unordered_set<ivec2, STPHashivec2> UnionChunkDictionary;

		void reserve(const size_t totalNeighbour) {
			this->UnionChunk.reserve(totalNeighbour);
			this->UnionChunkDictionary.reserve(totalNeighbour);
		}

		void clear() noexcept {
			this->UnionChunk.clear();
			this->UnionChunkDictionary.clear();
		}

	} HeightmapPreconditionCache, ErosionPreconditionCache, ErosionUnionCache;

	//used by function to schedule chunk computation order to avoid neighbour collision
	struct STPChunkScheduleCache {
	public:

		//all scheduled chunks including neighbours, serve as a lookup table
		unordered_multimap<ivec2, STPChunkPriorityNumber, STPHashivec2> ChunkSchdule;
		//see below, it is just a cache to be transferred to the actual memory to avoid reallocation.
		STPChunkSchedule::container_type CentreScheduleCache;
		//already scheduled centre chunks
		STPChunkSchedule CentreSchedule;
		//a temporary array for tracking priority numbers of all neighbours for the current chunk.
		vector<STPChunkPriorityNumber> CurrentChunkNeighbourPriority;
		//the dictionary for the above array to avoid duplication.
		unordered_set<STPChunkPriorityNumber> CurrentChunkNeighbourPriorityDictionary;

		void reserve(const STPChunk::STPChunkNeighbourOffset& offset, const size_t entryCount) {
			const size_t offsetCount = offset.NeighbourOffsetCount,
				offsetEntryProduct = offsetCount * entryCount;

			this->ChunkSchdule.reserve(offsetEntryProduct);
			this->CentreScheduleCache.reserve(entryCount);
			this->CurrentChunkNeighbourPriority.reserve(offsetCount);
			this->CurrentChunkNeighbourPriorityDictionary.reserve(offsetCount);
		}

		void clear() noexcept {
			this->ChunkSchdule.clear();
			this->CentreScheduleCache.clear();
			//by design of our algorithm all tasks should be completed so the schedule should be empty
			assert(this->CentreSchedule.empty());
			this->CurrentChunkNeighbourPriority.clear();
			this->CurrentChunkNeighbourPriorityDictionary.clear();
		}

	} ErosionScheduleCache;

	//used by function to request chunks
	struct {
	public:

		//corresponded entry in the chunk database for each request if they are not usable yet
		STPChunkDatabaseEntryCollection UnfinishedEntry;

		void reserve(const size_t entryCount) {
			this->UnfinishedEntry.reserve(entryCount);
		}

		void clear() noexcept {
			this->UnfinishedEntry.clear();
		}

	} ChunkRequestCache;
	
	//all generation functions share this future, and we guarantee the generations will be finished by the end of each function
	queue<future<void>> GenerationFuture;
	/* ----------------------------------------------------------------------------------------------------------------- */

	//all terrain map generators
	STPBiomeFactory& BiomemapGenerator;
	STPHeightfieldGenerator& HeightfieldGenerator;

public:

	STPTextureFactory& SplatmapGenerator;

private:

	STPThreadPool GeneratorWorker;

	/**
	 * @brief Calculate the offset of generated map such that the transition of each chunk is seamless.
	 * @param chunkCoord The world coordinate of the chunk.
	 * @return The chunk offset in world coordinate.
	*/
	inline dvec2 calcMapOffset(const ivec2& chunkCoord) const noexcept {
		const STPEnvironment::STPChunkSetting& chk_config = this->ChunkSetting;
		return STPChunk::calcChunkMapOffset(chunkCoord, chk_config.ChunkSize, chk_config.MapSize, chk_config.MapOffset);
	}

	/**
	 * @brief Calculate the relative offset of all neighbours regarding a centre chunk.
	 * @param region_size The size of the region.
	 * @return The chunk neighbour offset.
	*/
	inline STPChunk::STPChunkNeighbourOffset calcNeighbourOffset(const ivec2 region_size) const {
		return STPChunk::calcChunkNeighbourOffset(this->ChunkSetting.ChunkSize, region_size);
	}

	/**
	 * @brief Block and wait for all tasks in the generation task queue.
	*/
	void waitAllGenerationTasks() {
		while (!this->GenerationFuture.empty()) {
			//wait for the task to finish then pop
			this->GenerationFuture.front().get();
			this->GenerationFuture.pop();
		}
	}

	/**
	 * @brief Find the union of all neighbours of given chunks.
	 * @tparam Pred A function to decide if the current neighbour should be merged.
	 * @param chunk_entry A range of chunks to be merged.
	 * @param neighbour_offset The range of neighbour offset for each chunk.
	 * @param union_cache The cache memory for storing intermediate data for this operation
	 * to avoid repetitive memory allocation.
	 * @param predicate For a input chunk, return true to put this chunk into the union.
	 * Otherwise it will be ignored.
	 * @return A union of the input chunk entry with no duplicate.
	*/
	template<class Pred>
	const STPChunkDatabaseEntryCollection& unionChunkNeighbour(const STPChunkDatabaseEntryCollection& chunk_entry,
		const STPChunk::STPChunkNeighbourOffset& neighbour_offset, STPNeighbourUnionCache& union_cache, Pred&& predicate) {
		union_cache.clear();
		auto& [chunk_neighbour_union, chunk_neighbour_union_dict] = union_cache;
		
		//for each generating chunk, and for each of their neighbours...
		for (const auto entry : chunk_entry) {
			const ivec2 chunk_coord = entry->first;
			//find each neighbour from the database
			for (const auto neighbour_coord_offset : neighbour_offset) {
				//convert offset to the absolute coordinate of a chunk
				const ivec2 neighbour_coord = chunk_coord + neighbour_coord_offset;
				
				//find chunk from the database, or create a new one
				auto& curr_db_entry = *this->ChunkDatabase.try_emplace(neighbour_coord, this->ChunkSetting.MapSize).first;
				//status checking
				if (std::forward<Pred>(predicate)(curr_db_entry) && chunk_neighbour_union_dict.emplace(neighbour_coord).second) {
					//chunk passes the predicate and it is a newly encountered neighbour
					chunk_neighbour_union.emplace_back(&curr_db_entry);
				}
			}
		}
		assert(chunk_neighbour_union.size() == chunk_neighbour_union_dict.size());
		return chunk_neighbour_union;
	}

	//An overload without predicate, and it finds the union of all chunk neighbours
	//@see unionChunkNeighbour
	const STPChunkDatabaseEntryCollection& unionChunkNeighbour(const STPChunkDatabaseEntryCollection& ce,
		const STPChunk::STPChunkNeighbourOffset& no, STPNeighbourUnionCache& uc) {
		return this->unionChunkNeighbour(ce, no, uc, [](const auto&) { return true; });
	}

	/**
	 * @brief Attempt to verify preconditions for neighbours of all given chunks in any generation process, given an expected state.
	 * @param chunk_entry A range of chunks to be checked.
	 * @param neighbour_offset The range of neighbour offset for each chunk.
	 * @param expected_state The state for each neighbour for which, if the chunk state is less than this value, it fails the verification.
	 * As chunk state grows progressively, a state that is no less than will be treated as pass.
	 * @param precond_cache The cache memory for storing the intermediate data for this verification process
	 * to avoid repetitive memory allocation.
	 * @return A subset of the input chunk entry that does not meet the requirement.
	*/
	const STPChunkDatabaseEntryCollection& verifyNeighbourPrecondition(const STPChunkDatabaseEntryCollection& chunk_entry,
		const STPChunk::STPChunkNeighbourOffset& neighbour_offset, const STPChunk::STPChunkCompleteness expected_state, STPNeighbourUnionCache& precond_cache) {
		//put chunk into the union if it has not met the expected status
		return this->unionChunkNeighbour(chunk_entry, neighbour_offset, precond_cache,
			[expected_state](const auto& chunk) { return chunk.second.Completeness < expected_state; });
	}

	/**
	 * @brief Schedule the parallel computation order for a given array of chunks to avoid possible neighbour overlap thus race condition.
	 * Because each centre chunk is assigned with one thread, and neighbour chunks may overlap potentially,
	 * for example 2 centre chunks are right next to each other so their neighbours overlap.
	 * @param chunk_entry The range of chunk to be scheduled.
	 * @param neighbour_offset The range of neighbour offset for each chunk.
	 * @param schedule_cache The cache memory for storing intermediate data for chunk scheduling.
	 * @return The schedule of each chunk.
	 * @see STPChunkSchedule
	*/
	static STPChunkSchedule& scheduleChunk(const STPChunkDatabaseEntryCollection& chunk_entry, const STPChunk::STPChunkNeighbourOffset& neighbour_offset,
		STPChunkScheduleCache& schedule_cache) {
		schedule_cache.clear();
		auto& [chunk_schedule, centre_schedule_cache, centre_schedule, current_neighbour_priority,
			current_neighbour_priority_dict] = schedule_cache;

		//for every chunk, find their neighbour
		for (const auto entry : chunk_entry) {
			//clear priority for previous neighbours
			current_neighbour_priority.clear();
			current_neighbour_priority_dict.clear();

			const ivec2 chunk_coord = entry->first;
			//find the union of priority numbers in this neighbourhood
			//and reduce the union of priority number to a sorted unique range
			for (const auto offset : neighbour_offset) {
				const auto [begin, end] = chunk_schedule.equal_range(chunk_coord + offset);
				//insert into the union only when it is not a duplicate member
				for (auto it = begin; it != end; it++) {
					if (const STPChunkPriorityNumber pri = it->second;
						current_neighbour_priority_dict.emplace(pri).second) {
						//not a duplicate member
						current_neighbour_priority.emplace_back(pri);
					}
				}
			}
			std::sort(current_neighbour_priority.begin(), current_neighbour_priority.end());

			//now we need to determine the priority number for this range of chunks
			//as the range is sorted, we can read the smallest priority
			STPChunkPriorityNumber schedulePri = STPGeneratorManager::DummyChunkPriorityNumber;
			if (current_neighbour_priority.empty()) {
				//if there is no schedule in this neighbour, use the minimum priority
				schedulePri = STPGeneratorManager::MinimumChunkPriorityNumber;

				//else, what we want is a priority number that is not seen in the entire neighbourhood
			} else if (const STPChunkPriorityNumber minNeighbourPri = current_neighbour_priority.front();
				minNeighbourPri > STPGeneratorManager::MinimumChunkPriorityNumber) {
				//we can now pick a number that is just less than the minimum of the range
				schedulePri = minNeighbourPri - 1u;
			} else {
				//this is now a bit tricky
				//The priority should not go below the minimum allowed priority range.
				//We try to find non-consecutive priority in this range, so there is a gap between 2 numbers.
				//If we cannot find such gap, use one plus the max priority at the back of the range.
				//The container is sorted in non-decreasing order, and each element is unique,
				//so the latter number must be strictly greater than the prior number.
				const auto it = std::adjacent_find(current_neighbour_priority.begin(), current_neighbour_priority.end(),
					[](const auto a, const auto b) { return static_cast<STPChunkPriorityNumber>(b - a) > 1u; });
				schedulePri = it != current_neighbour_priority.end() ? *it + 1u : current_neighbour_priority.back() + 1u;
			}

			//record the new priority for this chunk and its neighbour
			centre_schedule_cache.emplace_back(schedulePri, entry);
			//record the priority for every chunks in this neighbourhood
			for_each(neighbour_offset.begin(), neighbour_offset.end(), [&cs = chunk_schedule, schedulePri, chunk_coord](const auto offset) {
				cs.emplace(chunk_coord + offset, schedulePri);
			});
		}

		//return with sorted container
		//we make a copy from the cache since we want to preserve the allocated memory
		//allocating memory is way more expensive than copying small and trivial data
		centre_schedule = STPChunkSchedule(STPChunkScheduleComparator { }, centre_schedule_cache);
		return centre_schedule;
	}

	/**
	 * @brief Generate biomemap for all given chunks.
	 * Precondition: chunk must be empty.
	 * @param chunk_entry The entries of chunks from the database to be generated.
	*/
	void generateBiomemap(const STPChunkDatabaseEntryCollection& chunk_entry) {
		const auto computeBiomemap = [this](STPChunkDatabaseEntry& chunk_entry) -> void {
			auto& [chunkCoord, chunk] = chunk_entry;
			//calculate map offset from world coordinate
			const ivec2 biomemap_offset = static_cast<ivec2>(glm::roundEven(this->calcMapOffset(chunkCoord)));
			//call biomemap generator
			this->BiomemapGenerator(chunk.biomemap(), biomemap_offset);

			//update chunk status
			chunk.Completeness = STPChunk::STPChunkCompleteness::BiomemapReady;
		};

		//start biomemap generation tasks
		for_each(chunk_entry.cbegin(), chunk_entry.cend(), [&tp = this->GeneratorWorker, &computer = computeBiomemap,
			&task_queue = this->GenerationFuture](const auto entry) {
			task_queue.emplace(tp.enqueue(computer, std::ref(*entry)));
		});
		//wait for all working threads to finish
		this->waitAllGenerationTasks();
	}

	/**
	 * @brief Generate heightmap for all given chunks.
	 * Precondition: chunk must be either empty or has biomemap computed.
	 * @param chunk_entry The entries of chunks from the database to be generated.
	*/
	void generateHeightmap(const STPChunkDatabaseEntryCollection& chunk_entry) {
		//heightmap generation requires use of biomemap
		//check if all neighbour chunks of the entry have biomemap ready
		if (const STPChunkDatabaseEntryCollection& chunk_biomemap_entry = this->verifyNeighbourPrecondition(chunk_entry, this->DiversityNeightbourOffset,
				STPChunk::STPChunkCompleteness::BiomemapReady, this->HeightmapPreconditionCache);
			!chunk_biomemap_entry.empty()) {
			//generate biomemap
			this->generateBiomemap(chunk_biomemap_entry);
		}

		//Precondition: chunk must have biomemap generated
		const auto computeHeightmap = [this](STPChunkDatabaseEntry& centre_chunk) -> void {
			auto& [chunkCoord, chunk] = centre_chunk;
			const STPChunk::STPChunkNeighbourOffset& neighbour_offset = this->DiversityNeightbourOffset;

			const auto& db = this->ChunkDatabase;
			//get the memory of all neighbour
			//since biomemap is read only, we can safely read from the database and get the biomemap memory,
			//even if some neighbours may overlap and be read by multiple threads
			unique_ptr<void*[]> neighbour_biome_mem = this->DiversityNeighbourMemoryPool.requestObject();
			const Sample** const neighbour_biome = const_cast<const Sample**>(reinterpret_cast<Sample**>(neighbour_biome_mem.get()));
			transform(neighbour_offset.begin(), neighbour_offset.end(), neighbour_biome,
				[&db, centre_chunk = chunkCoord](const auto chunk_coord_offset) {
					return db.find(centre_chunk + chunk_coord_offset)->second.biomemap();
				});
			//invoke generator
			this->HeightfieldGenerator.generate(chunk.heightmap(), neighbour_biome, static_cast<vec2>(this->calcMapOffset(chunkCoord)));
			//return memory
			this->DiversityNeighbourMemoryPool.returnObject(std::move(neighbour_biome_mem));

			chunk.Completeness = STPChunk::STPChunkCompleteness::HeightmapReady;
		};

		//start heightmap generation tasks
		for (const auto entry : chunk_entry) {
			this->GenerationFuture.emplace(this->GeneratorWorker.enqueue(computeHeightmap, std::ref(*entry)));
		}
		this->waitAllGenerationTasks();
	}

	/**
	 * @brief Erode heightmap for all given chunks.
	 * Precondition: centre chunk must not be complete.
	 * @param chunk_entry The entries of chunks from the database to be eroded.
	 * @return An array of chunks whose heightmap have been modified besides input chunk entry (input chunks are also included in this array)
	 * due to the nearest neighbour logic used during erosion algorithm.
	 * Application should reload heightmap of all chunks in the returned array.
	*/
	const STPChunkDatabaseEntryCollection& erodeHeightmap(const STPChunkDatabaseEntryCollection& chunk_entry) {
		//eroding a heightmap, well obviously, requires all neighbours to at least have heightmap available.
		//but neighbour can either be complete, or just having heightmap available.
		if (const STPChunkDatabaseEntryCollection& chunk_heightmap_entry = this->verifyNeighbourPrecondition(chunk_entry, this->ErosionNeighbourOffset,
				STPChunk::STPChunkCompleteness::HeightmapReady, this->ErosionPreconditionCache);
			!chunk_heightmap_entry.empty()) {
			//generate heightmap
			this->generateHeightmap(chunk_heightmap_entry);
		}
		//before doing the actual generation, we need to resolve the possible collision when writing to the neighbour chunks
		STPChunkSchedule& erosion_schedule = STPGeneratorManager::scheduleChunk(chunk_entry, this->ErosionNeighbourOffset, this->ErosionScheduleCache);

		//Precondition: chunk must have heightmap generated at least
		const auto computeHeightmapErosion = [this](STPChunkDatabaseEntry& chunk_entry) -> void {
			auto& [chunkCoord, chunk] = chunk_entry;
			const STPChunk::STPChunkNeighbourOffset& neighbour_offset = this->ErosionNeighbourOffset;
			
			//get memory, we need to 2 neighbour sets
			unique_ptr<void*[]> neighbour_heightmap_mem = this->ErosionNeighbourMemoryPool.requestObject(),
				neighbour_heightmap_low_mem = this->ErosionNeighbourMemoryPool.requestObject();
			float** const neighbour_heightmap = reinterpret_cast<float**>(neighbour_heightmap_mem.get());
			unsigned short** const neighbour_heightmap_low = reinterpret_cast<unsigned short**>(neighbour_heightmap_low_mem.get());
			//prepare pointers
			for (unsigned int local_idx = 0u; local_idx < neighbour_offset.NeighbourOffsetCount; local_idx++) {
				STPChunk& curr_chunk = this->ChunkDatabase.find(chunkCoord + neighbour_offset[local_idx])->second;

				neighbour_heightmap[local_idx] = curr_chunk.heightmap();
				neighbour_heightmap_low[local_idx] = curr_chunk.heightmapLow();
			}
			//where the magic happens
			this->HeightfieldGenerator.erode(neighbour_heightmap, neighbour_heightmap_low);
			//clean up
			this->ErosionNeighbourMemoryPool.returnObject(std::move(neighbour_heightmap_mem));
			this->ErosionNeighbourMemoryPool.returnObject(std::move(neighbour_heightmap_low_mem));

			chunk.Completeness = STPChunk::STPChunkCompleteness::Complete;
		};

		//we submit tasks in batch, each batch should have the same priority number
		//start from the highest priority
		STPChunkPriorityNumber batchPriority = erosion_schedule.top().first;
		while (!erosion_schedule.empty()) {
			const auto [chunkPri, chunkEntry] = erosion_schedule.top();
			if (chunkPri == batchPriority) {
				//launch a new task and pop
				this->GenerationFuture.emplace(this->GeneratorWorker.enqueue(computeHeightmapErosion, std::ref(*chunkEntry)));
				erosion_schedule.pop();
				continue;
			}

			//encounter a new batch, wait for all tasks in this batch to finish
			this->waitAllGenerationTasks();
			//go to the next batch
			batchPriority = chunkPri;
		}

		//tell application which chunks have changed, besides chunks specified in the input
		//because erosion may also modify neighbour chunks from the input
		return this->unionChunkNeighbour(chunk_entry, this->ErosionNeighbourOffset, this->ErosionUnionCache);
	}

public:

	//data for chunk requested
	//the chunk coordinate must be provided as data header, and pointer to chunk will be returned as response
	typedef vector<ivec2> STPChunkRequestPayload;
	//one data entry in the response based on the request
	typedef STPChunkDatabaseEntryCollection STPChunkRequestResponseEntry;
	//store the response from the last chunk request query
	struct STPChunkRequestResponse {
	public:

		//The response based on each entry in the original request
		STPChunkRequestResponseEntry PrimaryResponse;

		//The following additional responses indicate addition operations required besides the original request.
		//Chunks that required heightmap to be reloaded because this request may trigger heightmap modification
		//outside the requested chunks.
		//This can be a null if there is no such heightmap exists.
		const STPChunkRequestResponseEntry* RequireHeightmapReload;

		void reserve(const size_t entryCount) {
			this->PrimaryResponse.reserve(entryCount);
		}

		void clear() noexcept {
			this->PrimaryResponse.clear();
			this->RequireHeightmapReload = nullptr;
		}

	} LastChunkRequestResponse;

	/**
	 * @brief Initialise generator manager with pipeline stages filled with generators.
	 * @param setup The pointer to all pipeline stages.
	 * @param pipeline The pointer to the world pipeline registered with the generator manager.
	*/
	STPGeneratorManager(const STPWorldPipeline::STPPipelineSetup& setup, STPWorldPipeline& pipeline) :
		Pipeline(pipeline), ChunkSetting(this->Pipeline.ChunkSetting),
		DiversityNeightbourOffset(this->calcNeighbourOffset(this->ChunkSetting.DiversityNearestNeighbour)),
		ErosionNeighbourOffset(this->calcNeighbourOffset(this->ChunkSetting.ErosionNearestNeighbour)),
		DiversityNeighbourMemoryPool(this->ChunkSetting.DiversityNearestNeighbour), ErosionNeighbourMemoryPool(this->ChunkSetting.ErosionNearestNeighbour),
		BiomemapGenerator(*setup.BiomemapGenerator), HeightfieldGenerator(*setup.HeightfieldGenerator), SplatmapGenerator(*setup.SplatmapGenerator), 
		GeneratorWorker(5u) {
		//reserve memory for various cache
		//we consider the maximum number of possible nearest neighbour chunk for each generation stage
		constexpr static auto calcEntryCount = [](const uvec2& entry_range) constexpr -> size_t {
			return entry_range.x * entry_range.y;
		};
		const STPEnvironment::STPChunkSetting& chk = this->ChunkSetting;
		//minus 1, which is the centre chunk
		//rendering requires erosion of heightmap in render distance
		const uvec2 nn_erosion = chk.RenderDistance,
			//erosion requires heightmap from some neighbour chunks
			nn_heightmap = nn_erosion - 1u + chk.ErosionNearestNeighbour,
			//heightmap generation requires biomemap from some neighbour chunks
			nn_biomemap = nn_heightmap - 1u + chk.DiversityNearestNeighbour;
		const size_t erosionCount = calcEntryCount(nn_erosion),
			heightmapCount = calcEntryCount(nn_heightmap),
			biomemapCount = calcEntryCount(nn_biomemap);
		
		this->ErosionUnionCache.reserve(heightmapCount);
		this->ErosionPreconditionCache.reserve(heightmapCount);
		this->HeightmapPreconditionCache.reserve(biomemapCount);

		this->ErosionScheduleCache.reserve(this->ErosionNeighbourOffset, erosionCount);

		this->ChunkRequestCache.reserve(erosionCount);
		this->LastChunkRequestResponse.reserve(erosionCount);
	}

	STPGeneratorManager(const STPGeneratorManager&) = delete;

	STPGeneratorManager(STPGeneratorManager&&) = delete;

	STPGeneratorManager& operator=(const STPGeneratorManager&) = delete;

	STPGeneratorManager& operator=(STPGeneratorManager&&) = delete;

	~STPGeneratorManager() {
		this->waitAllGenerationTasks();
	}

	/**
	 * @brief Request chunk data.
	 * All generation will be handled automatically by the generator manager, if chunk is not available.
	 * All chunks are guaranteed to be available when this function returns.
	 * @param request The chunk world coordinate of chunk requested.
	 * @return The response based on the request.
	 * The response result will be available until the next initiation of request.
	*/
	const STPChunkRequestResponse& requestChunk(const STPChunkRequestPayload& request) {
		const size_t requestCount = request.size();
		//request data
		this->ChunkRequestCache.clear();
		auto& [generation_entry] = this->ChunkRequestCache;
		//response data
		this->LastChunkRequestResponse.clear();
		auto& [response_primary, response_heightmap_reload] = this->LastChunkRequestResponse;
		
		//check if any chunks are ready to use, if not, prepare for generation
		for (const auto chunk_coord : request) {
			if (const auto entry = response_primary.emplace_back(
					&(*this->ChunkDatabase.try_emplace(chunk_coord, this->ChunkSetting.MapSize).first));
				entry->second.Completeness != STPChunk::STPChunkCompleteness::Complete) {
				//need to generate
				generation_entry.emplace_back(entry);
			}
		}
		if (!generation_entry.empty()) {
			//invoke generators, start from the top of the generation chain
			response_heightmap_reload = &this->erodeHeightmap(generation_entry);
		}

		return this->LastChunkRequestResponse;
	}

	/**
	 * @brief Prepare and generate splatmap for rendering.
	 * @tparam M The terrain map memory.
	 * @param buffer The terrain map on device side, a mapped OpenGL pointer.
	 * @param requesting_chunk Specify chunks that need to have splatmap generated.
	 * Note that the coordinate of local chunk should specify chunk map offset rather than chunk world position.
	*/
	template<class M>
	void generateSplatmap(const M& buffer,
		const STPTextureFactory::STPRequestingChunkInfo& requesting_chunk) const {
		//prepare texture and surface object
		STPSmartDeviceObject::STPTexture biomemap, heightfield;
		STPSmartDeviceObject::STPSurface splatmap;
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
		biomemap = STPSmartDeviceObject::makeTexture(res_desc, tex_desc, nullptr);

		//heightfield
		res_desc.res.array.array = buffer[TERRAIN_MAP_INDEX(STPWorldPipeline::STPTerrainMapType::Heightmap)];
		tex_desc.filterMode = cudaFilterModeLinear;
		tex_desc.readMode = cudaReadModeNormalizedFloat;
		heightfield = STPSmartDeviceObject::makeTexture(res_desc, tex_desc, nullptr);

		//splatmap
		res_desc.res.array.array = buffer[TERRAIN_MAP_INDEX(STPWorldPipeline::STPTerrainMapType::Splatmap)];
		splatmap = STPSmartDeviceObject::makeSurface(res_desc);

		const cudaStream_t stream = this->Pipeline.BufferStream.get();
		//launch splatmap computer
		this->SplatmapGenerator(biomemap.get(), heightfield.get(), splatmap.get(), requesting_chunk, stream);

		//before deallocation happens make sure everything has finished.
		STP_CHECK_CUDA(cudaStreamSynchronize(stream));
	}

};

/**
 * @brief STPConcurrentStreamManager handles multiple streams and their synchronisation to exploit stream parallelism.
*/
class STPConcurrentStreamManager {
private:

	//Specifies how many parallel worker should be used.
	constexpr static unsigned char Parallelism = 4u;
	//Locate the currently active memory worker.
	unsigned char WorkerIndex = 0u;

	//The transfer cache is used because cudaMemcpy2DArrayToArray does not have a streamed version.
	//Each worker will have a transfer cache.
	array<STPSmartDeviceMemory::STPPitchedDeviceMemory<unsigned char[]>, Parallelism> MapTransferCache;
	//N auxiliary workers
	array<STPSmartDeviceObject::STPStream, Parallelism> MemoryWorker;
	//synchronisation of N workers + the main working stream
	array<STPSmartDeviceObject::STPEvent, Parallelism + 1u> MemoryWorkerSync;

public:

	/**
	 * @brief Initialise a concurrent stream manager instance.
	 * @param cache_texture_dim Species the dimension of the cache texture.
	 * Each stream will be allocated with one cache texture.
	*/
	STPConcurrentStreamManager(const uvec2 cache_texture_dim) {
		using std::generate;
		using std::bind;
		//create stream and event
		generate(this->MemoryWorker.begin(), this->MemoryWorker.end(),
			bind(static_cast<STPSmartDeviceObject::STPStream (&)(unsigned int)>(STPSmartDeviceObject::makeStream), cudaStreamNonBlocking));
		generate(this->MemoryWorkerSync.begin(), this->MemoryWorkerSync.end(), bind(STPSmartDeviceObject::makeEvent, cudaEventDisableTiming));
		//allocate memory for transfer cache for each stream
		generate(this->MapTransferCache.begin(), this->MapTransferCache.end(), [&dim = cache_texture_dim]() {
			return STPSmartDeviceMemory::makePitchedDevice<unsigned char[]>(dim.x, dim.y);
		});
	}

	STPConcurrentStreamManager(const STPConcurrentStreamManager&) = delete;

	STPConcurrentStreamManager(STPConcurrentStreamManager&&) = delete;

	STPConcurrentStreamManager& operator=(const STPConcurrentStreamManager&) = delete;

	STPConcurrentStreamManager& operator=(STPConcurrentStreamManager&&) = delete;

	~STPConcurrentStreamManager() {
		//wait for all worker streams to finish
		for_each(this->MemoryWorker.cbegin(), this->MemoryWorker.cend(),
			[](const auto& stream) { STP_CHECK_CUDA(cudaStreamSynchronize(stream.get())); });
	}

	/**
	 * @brief Ask the worker streams to wait for the main stream before starting tasks enqueued after.
	 * @param main The main stream whom workers are waiting for.
	*/
	inline void workersWaitMain(const cudaStream_t main) const {
		//the last event is for the main stream
		const cudaEvent_t mainEvent = this->MemoryWorkerSync[STPConcurrentStreamManager::Parallelism].get();

		//record the status of the main stream
		STP_CHECK_CUDA(cudaEventRecord(mainEvent, main));
		//all workers thread should not begin later tasks until the main stream has reached this point
		for_each(this->MemoryWorker.cbegin(), this->MemoryWorker.cend(),
			[&mainEvent](const auto& stream) { STP_CHECK_CUDA(cudaStreamWaitEvent(stream.get(), mainEvent)); });
	}

	/**
	 * @brief Ask the main stream to wait for all worker streams before starting tasks enqueued after.
	 * @param main The main stream to wait.
	*/
	inline void mainWaitsWorkers(const cudaStream_t main) const {
		//record the status of all worker streams.
		for (size_t worker = 0u; worker < this->MemoryWorker.size(); worker++) {
			STP_CHECK_CUDA(cudaEventRecord(this->MemoryWorkerSync[worker].get(), this->MemoryWorker[worker].get()));
		}
		//main stream should not continue until all workers are done
		for_each(this->MemoryWorkerSync.cbegin(), this->MemoryWorkerSync.cend(),
			[&main](const auto& event) { STP_CHECK_CUDA(cudaStreamWaitEvent(main, event.get())); });
	}

	/**
	 * @brief Grab the next memory worker stream.
	 * We cycle the workers to maximise ability of parallelism.
	 * @return The currently active memory worker.
	*/
	inline cudaStream_t nextWorker() noexcept {
		const unsigned char idx = this->WorkerIndex++;
		//wrap the index around
		this->WorkerIndex %= STPConcurrentStreamManager::Parallelism;
		return this->MemoryWorker[idx].get();
	}

	/**
	 * @brief Get the transfer cache of the current memory worker.
	 * @return The pointer to the transfer cache.
	*/
	inline auto& currentTransferCache() noexcept {
		//worker 0 uses cache 1, 1 uses 2, ..., N - 2 uses N - 1, N - 1 uses 0.
		return this->MapTransferCache[this->WorkerIndex];
	}

};

#define FOR_EACH_MEMORY() for (size_t i = 0u; i < STPWorldPipeline::STPMemoryManager::MemoryUnitCapacity; i++)

//world pipeline: min chunk coordinate
constexpr static ivec2 wpMinChunkCoordinate = ivec2(numeric_limits<int>::min());

class STPWorldPipeline::STPMemoryManager {
public:

	//The number of map data in a memory unit.
	constexpr static size_t MemoryUnitCapacity = 3u;
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

		//from the memory manager
		const STPChunk::STPChunkNeighbourOffset& LocalChunkOffsetTable;

		//terrain map but the memory is automatically managed.
		STPMemoryUnit<STPSmartDeviceObject::STPGLTextureObject> TerrainMapManaged;
		//a shallow copy to the pointer to the managed terrain map.
		//index 0: R16UI biome map
		//index 1: R16 height map
		//index 2: R8UI splat map
		STPMemoryUnit<GLuint> TerrainMap;
		//managed graphics resource handle.
		STPMemoryUnit<STPSmartDeviceObject::STPGraphicsResource> TerrainMapResourceManaged;
		//registered CUDA graphics handle for GL terrain maps, copy of the managed memory.
		STPMemoryUnit<cudaGraphicsResource_t> TerrainMapResource;

		const uvec2 ChunkSize, RenderDistance;
		//The origin (top-left chunk) of the current render distance.
		ivec2 LocalOriginCoordinate;
		//Vector that stored rendered chunk world position and loading status (True is loaded, false otherwise).
		vector<pair<ivec2, bool>> LocalChunkRecord;

		/**
		 * @brief Initialise a memory block.
		 * @param manager The pointer to the dependent world memory manager.
		*/
		STPMemoryBlock(const STPMemoryManager& manager) : LocalChunkOffsetTable(manager.RenderChunkNeighbourOffset),
			ChunkSize(manager.ChunkSetting.ChunkSize), RenderDistance(manager.ChunkSetting.RenderDistance),
			LocalOriginCoordinate(wpMinChunkCoordinate) {
			const STPEnvironment::STPChunkSetting& setting = manager.ChunkSetting;
			const auto setupMap = [buffer_size = setting.RenderDistance * setting.MapSize](const GLuint texture,
				const GLint min_filter, const GLint mag_filter, const GLenum internal) -> void {
				//set texture parameter
				glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
				glTextureParameteri(texture, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
				glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, min_filter);
				glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, mag_filter);
				//allocation
				glTextureStorage2D(texture, 1, internal, buffer_size.x, buffer_size.y);
			};
			//graphics register flag for each corresponding texture
			constexpr static STPMemoryUnit<unsigned int> registerFlag = {
				cudaGraphicsRegisterFlagsNone,
				cudaGraphicsRegisterFlagsNone,
				cudaGraphicsRegisterFlagsSurfaceLoadStore
			};

			using std::generate;
			//creating texture map and copy the raw handles into a separate array, so we can access them easily
			generate(this->TerrainMapManaged.begin(), this->TerrainMapManaged.end(), []() {
				return STPSmartDeviceObject::makeGLTextureObject(GL_TEXTURE_2D);
			});
			transform(this->TerrainMapManaged.cbegin(), this->TerrainMapManaged.cend(), this->TerrainMap.begin(),
				[](const auto& managed_texture) { return managed_texture.get(); });
			//biomemap
			setupMap(this->TerrainMap[0], GL_NEAREST, GL_NEAREST, GL_R16UI);
			//heightfield
			setupMap(this->TerrainMap[1], GL_LINEAR, GL_LINEAR, GL_R16);
			//splatmap
			setupMap(this->TerrainMap[2], GL_NEAREST, GL_NEAREST, GL_R8UI);

			//create CUDA mapping of the texture
			transform(this->TerrainMap.cbegin(), this->TerrainMap.cend(), registerFlag.cbegin(), this->TerrainMapResourceManaged.begin(),
				[](const auto tbo, const auto flag) {
					return STPSmartDeviceObject::makeGLImageResource(tbo, GL_TEXTURE_2D, flag);
				});
			transform(this->TerrainMapResourceManaged.cbegin(), this->TerrainMapResourceManaged.cend(),
				this->TerrainMapResource.begin(), [](const auto& res) { return res.get(); });

			//allocate memory for lookup tables
			this->LocalChunkRecord.resize(this->LocalChunkOffsetTable.NeighbourOffsetCount);
		}

		STPMemoryBlock(const STPMemoryBlock&) = delete;

		STPMemoryBlock(STPMemoryBlock&&) = delete;

		STPMemoryBlock& operator=(const STPMemoryBlock&) = delete;

		STPMemoryBlock& operator=(STPMemoryBlock&&) = delete;

		~STPMemoryBlock() = default;

		/**
		 * @brief Calculate the index coordinate of this chunk in the current rendering distance.
		 * @param chunk_coord The chunk world coordinate.
		 * @return The chunk local index.
		*/
		inline ivec2 calcLocalIndexCoordinate(const ivec2& chunk_coord) const noexcept {
			return (chunk_coord - this->LocalOriginCoordinate) / static_cast<ivec2>(this->ChunkSize);
		}

		/**
		 * @brief Convert the chunk local index coordinate to the local index.
		 * The result is correct only if the chunk is in the rendering distance.
		 * @param localIdxCoord The chunk local index coordinate.
		 * @return The chunk local index.
		*/
		inline unsigned int toLocalIndex(const ivec2& localIdxCoord) const noexcept {
			//the 2D index will always be a positive number
			//convert to 1D index
			return localIdxCoord.x + localIdxCoord.y * this->RenderDistance.x;
		}

		/**
		 * @brief Check if this chunk's local index coordinate is in the current rendering distance.
		 * @param localIdxCoord The chunk local index coordinate.
		 * @return True if the chunk is in the rendering distance.
		*/
		inline bool isLocalChunk(const ivec2& localIdxCoord) const noexcept {
			const ivec2 iRenderDistance = static_cast<ivec2>(this->RenderDistance);
			//might be negative if it is outside the local
			return localIdxCoord.x >= 0 && localIdxCoord.x < iRenderDistance.x
				&& localIdxCoord.y >= 0 && localIdxCoord.y < iRenderDistance.y;
		}

		/**
		 * @brief Set the CUDA resource mapping flag for each GL texture.
		 * @param flag The resource mapping flag.
		*/
		inline void setMappingFlag(const unsigned int flag) const {
			FOR_EACH_MEMORY() {
				STP_CHECK_CUDA(cudaGraphicsResourceSetMapFlags(this->TerrainMapResource[i], flag));
			}
		}

		/**
		 * @brief Recompute the chunk tables.
		 * @param chunk_coord The chunk local coordinate of the centre of the range of chunks.
		*/
		inline void recomputeLocalChunkTable(const ivec2& centre_chunk_coord) {
			//no need to clear, we overwrite all elements
			transform(this->LocalChunkOffsetTable.begin(), this->LocalChunkOffsetTable.end(), this->LocalChunkRecord.begin(),
				[centre_chunk_coord](const auto chunk_offset) { return make_pair(centre_chunk_coord + chunk_offset, false); });

			//recompute the local origin
			this->LocalOriginCoordinate = STPChunk::calcLocalChunkOrigin(centre_chunk_coord, this->ChunkSize, this->RenderDistance);
		}

		/**
		 * @brief Map the GL terrain texture to CUDA array.
		 * @param stream The CUDA stream where the work is submitted to.
		 * @return The mapped CUDA array.
		*/
		inline STPMappedMemoryUnit mapTerrainMap(const cudaStream_t stream) {
			STPMappedMemoryUnit map;
			//map the texture, all OpenGL related work must be done on the main contexted thread
			//CUDA will make sure all previous graphics API calls are finished before stream begins
			STP_CHECK_CUDA(cudaGraphicsMapResources(3, this->TerrainMapResource.data(), stream));
			//we only have one texture, so index is always zero
			FOR_EACH_MEMORY() {
				STP_CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&map[i], this->TerrainMapResource[i], 0, 0));
			}

			return map;
		}

		/**
		 * @brief Unmap the GL terrain texture from CUDA array.
		 * @param stream The CUDA stream where the unmap should be synchronised.
		*/
		inline void unmapTerrainmap(const cudaStream_t stream) {
			//sync the stream that modifies the texture and unmap the chunk
			//CUDA will make sure all previous pending works in the stream has finished before graphics API can be called
			STP_CHECK_CUDA(cudaGraphicsUnmapResources(3, this->TerrainMapResource.data(), stream));
		}

	};

private:

	const cudaStream_t PipelineStream;
	const STPEnvironment::STPChunkSetting& ChunkSetting;

	//Record indices of chunks that need to have splatmap computed for the current task.
	unordered_set<unsigned int> SplatmapComputeChunk;
	constexpr static array<ivec2, 9u> NeighbourCoordinateOffset = {
		ivec2(-1, -1),
		ivec2( 0, -1),
		ivec2(+1, -1),
		ivec2(-1,  0),
		ivec2( 0,  0),
		ivec2(+1,  0),
		ivec2(-1, +1),
		ivec2( 0, +1),
		ivec2(+1, +1)
	};
	//using the centre chunk as the origin, the relative offsets of all chunks in a render area
	const STPChunk::STPChunkNeighbourOffset RenderChunkNeighbourOffset;

	array<STPMemoryBlock, 2u> MemoryBuffer;
	//clear buffer is just a memory filled with the same value as clear value.
	STPSmartDeviceMemory::STPPitchedDeviceMemory<unsigned char[]> MapClearBuffer;
	//Store the CUDA array after mapping the back buffer
	STPMemoryBlock::STPMappedMemoryUnit MappedBackBuffer;
	bool isBackBufferMapped;

	/**
	 * @brief Calculate the map offset on the terrain map to reach a specific local chunk.
	 * @param index The local chunk index.
	 * @return The chunk memory offset on the map.
	*/
	inline uvec2 calcLocalMapOffset(const unsigned int index) const noexcept {
		const STPEnvironment::STPChunkSetting& chunk_setting = this->ChunkSetting;
		const uvec2 chunkIdx = STPChunk::calcLocalChunkCoordinate(index, chunk_setting.RenderDistance);
		return chunk_setting.MapSize * chunkIdx;
	}

	/**
	 * @brief Calculate the size of cache memory for each chunk.
	 * @return The cache size, x is in the number of byte while y is the number of element.
	*/
	inline uvec2 calcChunkCacheSize() const noexcept {
		return this->ChunkSetting.MapSize * uvec2(STPMemoryManager::MaxMemoryFormat, 1u);
	}

public:

	/* -------------------------------- memory cache --------------------------------- */
	//array of chunks to have splatmap generated
	STPDiversity::STPTextureFactory::STPRequestingChunkInfo SplatmapInfoCache;
	//chunks to request from the generator manager
	STPGeneratorManager::STPChunkRequestPayload GeneratorRequestPayload;

	//the result of intersection between a given set of chunks and the render distance
	struct STPRenderDistanceIntersectionCache {
	public:

		STPGeneratorManager::STPChunkRequestResponseEntry ValidIntersection;

		void reserve(const size_t renderCount) {
			this->ValidIntersection.reserve(renderCount);
		}

		void clear() noexcept {
			this->ValidIntersection.clear();
		}

	} HeightmapIntersectionCache;
	/* ------------------------------------------------------------------------------- */

	STPConcurrentStreamManager Worker;

	//Double buffering.
	//Front buffer is passed to the user.
	const STPMemoryBlock* FrontBuffer;
	//Back buffer holds the currently in-progress generation.
	STPMemoryBlock* BackBuffer;

private:

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
		PipelineStream(pipeline.BufferStream.get()), ChunkSetting(pipeline.ChunkSetting),
		RenderChunkNeighbourOffset(STPChunk::calcChunkNeighbourOffset(this->ChunkSetting.ChunkSize, this->ChunkSetting.RenderDistance)),
		MemoryBuffer{ *this, *this }, isBackBufferMapped(false),
		Worker(this->calcChunkCacheSize()),
		FrontBuffer(&this->MemoryBuffer[0]), BackBuffer(&this->MemoryBuffer[1]) {
		const STPEnvironment::STPChunkSetting& setting = this->ChunkSetting;
		const uvec2 mapDim = setting.MapSize,
			clearBuffer_size = this->calcChunkCacheSize();

		//init clear buffers that are used to clear texture when new rendered chunks are loaded
		//(we need to clear the previous chunk data)
		this->MapClearBuffer =
			STPSmartDeviceMemory::makePitchedDevice<unsigned char[]>(clearBuffer_size.x, clearBuffer_size.y);

		//clear the `clear` buffer
		STP_CHECK_CUDA(cudaMemset2DAsync(this->MapClearBuffer.get(), this->MapClearBuffer.Pitch, 0x80u,
			clearBuffer_size.x, clearBuffer_size.y, this->PipelineStream));

		//initialise pixel value in the texture rather than leaving them as undefined.
		for (auto& block : this->MemoryBuffer) {
			const auto& mapped = block.mapTerrainMap(this->PipelineStream);

			//clear every chunk in parallel
			//make sure workers do not start until main stream has mapped the texture
			this->Worker.workersWaitMain(this->PipelineStream);
			for (unsigned int y = 0u; y < setting.RenderDistance.y; y++) {
				for (unsigned int x = 0u; x < setting.RenderDistance.x; x++) {
					const uvec2 localMapOffset = uvec2(x, y) * setting.MapSize;
					FOR_EACH_MEMORY() {
						STP_CHECK_CUDA(cudaMemcpy2DToArrayAsync(mapped[i], localMapOffset.x * MemoryFormat[i],
							localMapOffset.y, this->MapClearBuffer.get(), this->MapClearBuffer.Pitch,
							mapDim.x * MemoryFormat[i], mapDim.y, cudaMemcpyDeviceToDevice, this->Worker.nextWorker()));
					}
				}
			}
			//also ensure main stream does not unmap the texture while workers are working hard
			this->Worker.mainWaitsWorkers(this->PipelineStream);
			block.unmapTerrainmap(this->PipelineStream);
		}
		//clean up
		STP_CHECK_CUDA(cudaStreamSynchronize(this->PipelineStream));
		this->correctResourceMappingFlag();

		const unsigned int chunkCount = setting.RenderDistance.x * setting.RenderDistance.y;
		this->SplatmapComputeChunk.reserve(chunkCount);
		this->SplatmapInfoCache.reserve(chunkCount);
		this->GeneratorRequestPayload.reserve(chunkCount);
		this->HeightmapIntersectionCache.reserve(chunkCount);
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
	inline GLuint getMap(const STPWorldPipeline::STPTerrainMapType type) const noexcept {
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
		this->MappedBackBuffer = this->BackBuffer->mapTerrainMap(this->PipelineStream);
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
		this->BackBuffer->unmapTerrainmap(this->PipelineStream);
		this->isBackBufferMapped = false;
	}

	/**
	 * @brief Get the array to the mapped back buffer.
	 * @return The array to the mapped back buffer.
	*/
	inline const STPMemoryBlock::STPMappedMemoryUnit& getMappedBackBuffer() const noexcept {
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

		//map the front buffer as read-only
		STPMemoryBlock& frontBuffer_w = const_cast<STPMemoryBlock&>(*this->FrontBuffer);
		const STPMemoryBlock::STPMappedMemoryUnit frontBuffer_ptr = frontBuffer_w.mapTerrainMap(this->PipelineStream);

		//here we perform an optimisation: reuse chunk that has been rendered previously from the front buffer
		//make sure memory is available before any worker can begin
		this->Worker.workersWaitMain(this->PipelineStream);
		for (auto& [chunkPos, loaded] : this->BackBuffer->LocalChunkRecord) {
			const cudaStream_t stream_worker = this->Worker.nextWorker();

			//checking the new back buffer chunk, is there any old chunk has the same world coordinate as the new chunk?
			const unsigned int back_buffer_chunkIdx = this->BackBuffer->toLocalIndex(this->BackBuffer->calcLocalIndexCoordinate(chunkPos));
			if (const ivec2 front_buffer_chunkIdxCoord = this->FrontBuffer->calcLocalIndexCoordinate(chunkPos);
				this->FrontBuffer->isLocalChunk(front_buffer_chunkIdxCoord)) {
				//found, check if the previous cache is complete
				const unsigned int front_buffer_chunkIdx = this->FrontBuffer->toLocalIndex(front_buffer_chunkIdxCoord);

				if (this->FrontBuffer->LocalChunkRecord[front_buffer_chunkIdx].second) {
					//if the previous front buffer chunk is complete, copy to the back buffer
					{
						const uvec2 src_offset = this->calcLocalMapOffset(front_buffer_chunkIdx),
							dest_offset = this->calcLocalMapOffset(back_buffer_chunkIdx);

						//Each worker is assigned with a transfer cache, and works submitted to one worker queue are executed sequentially.
						const auto& trans_cache = this->Worker.currentTransferCache();
						FOR_EACH_MEMORY() {
							//front buffer -> cache
							STP_CHECK_CUDA(cudaMemcpy2DFromArrayAsync(trans_cache.get(), trans_cache.Pitch,
								frontBuffer_ptr[i], src_offset.x * MemoryFormat[i], src_offset.y,
								mapDim.x * MemoryFormat[i], mapDim.y, cudaMemcpyDeviceToDevice, stream_worker));
							//cache -> back buffer
							STP_CHECK_CUDA(cudaMemcpy2DToArrayAsync(backBuffer_ptr[i], dest_offset.x * MemoryFormat[i],
								dest_offset.y, trans_cache.get(), trans_cache.Pitch, mapDim.x * MemoryFormat[i],
								mapDim.y, cudaMemcpyDeviceToDevice, stream_worker));
						}
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
				FOR_EACH_MEMORY() {
					STP_CHECK_CUDA(cudaMemcpy2DToArrayAsync(backBuffer_ptr[i], dest_offset.x * MemoryFormat[i],
						dest_offset.y, this->MapClearBuffer.get(), this->MapClearBuffer.Pitch,
						mapDim.x * MemoryFormat[i], mapDim.y, cudaMemcpyDeviceToDevice, stream_worker));
				}
			}
		}
		//make sure workers on the memory are finished before releasing
		this->Worker.mainWaitsWorkers(this->PipelineStream);
		//unmap the front buffer
		frontBuffer_w.unmapTerrainmap(this->PipelineStream);

		//record indices of chunks that need to be computed
		this->SplatmapComputeChunk.clear();
		//To avoid artefacts, if any chunk is a neighbour of those chunks we just recorded, need to recompute them as
		//well. This is to mainly avoid splatmap seams. The logic is, if previously those chunks do not have a valid
		//neighbour but this time they have one, The border of them may not be aligned properly with the new chunks.
		for (const auto& [chunkCoord, chunkLoaded] : this->BackBuffer->LocalChunkRecord) {
			if (chunkLoaded) {
				//if this chunk already has data, skip it along with its neighbours.
				continue;
			}
			//for each neighbour of each computing chunk...
			for (const ivec2 coord_offset : this->NeighbourCoordinateOffset) {
				//make sure the coordinate is in a valid render distance
				if (const ivec2 current_coord = this->BackBuffer->calcLocalIndexCoordinate(chunkCoord + coord_offset);
					this->BackBuffer->isLocalChunk(current_coord)) {
					this->SplatmapComputeChunk.emplace(this->BackBuffer->toLocalIndex(current_coord));
				}
			}
		}
	}

	/**
	 * @brief Transfer terrain map on host side to device (OpenGL) texture by a local chunk.
	 * @param buffer Texture map on device side, a mapped OpenGL pointer.
	 * @param source The source of data.
	 * @param map_type The type of the terrain map.
	 * @param chunkIdx Local chunk index that specified which chunk in render area will be overwritten.
	 * @param stream The stream where works are sent.
	*/
	void sendChunkToBuffer(const STPMemoryBlock::STPMappedMemoryUnit& buffer, const void* const source,
		const STPTerrainMapType map_type, const unsigned int chunkIdx, const cudaStream_t stream) {
		const uvec2 dimension = this->ChunkSetting.MapSize,
			buffer_offset = this->calcLocalMapOffset(chunkIdx);
		const size_t index = TERRAIN_MAP_INDEX(map_type),
			channelSize = STPMemoryManager::MemoryFormat[index];
		
		//copy buffer to GL texture
		STP_CHECK_CUDA(cudaMemcpy2DToArrayAsync(buffer[index], buffer_offset.x * channelSize, buffer_offset.y, source,
			dimension.x * channelSize, dimension.x * channelSize, dimension.y, cudaMemcpyHostToDevice, stream));
	}

	/**
	 * @brief Recompute the local chunk status record and index lookup table.
	 * @param chunkPos The new chunk position.
	*/
	inline void recomputeLocalChunkTable(const ivec2& chunkPos) {
		this->BackBuffer->recomputeLocalChunkTable(chunkPos);
	}

	/**
	 * @brief Generate a list of chunks that are required to have splatmap computed, based on the compute table.
	 * Therefore splatmap of all chunks should be computed after all chunks have finished.
	 * @param force_regenerate An optional array of chunk entry which states the chunks in the array should have splatmap regenerated.
	 * It is undefined behaviour if any chunk is not in the rendering distance.
	 * @return The splatmap generator requesting info.
	*/
	const STPDiversity::STPTextureFactory::STPRequestingChunkInfo& generateSplatmapGeneratorInfo(
		const STPGeneratorManager::STPChunkRequestResponseEntry* force_regenerate) {
		using STPLocalChunkInformation = STPDiversity::STPTextureInformation::STPSplatGeneratorInformation::STPLocalChunkInformation;
		//given local chunk index
		const auto createLocalChunkInfo = [&chunk_setting = as_const(this->ChunkSetting), &localRec = as_const(this->BackBuffer->LocalChunkRecord)](
			const unsigned int index) -> STPLocalChunkInformation {
			//mark updated rendering buffer
			//we need to use the chunk normalised coordinate to get the splatmap offset,
			//splatmap offset needs to be consistent with the heightmap and biomemap
			const vec2 offset = static_cast<vec2>(STPChunk::calcChunkMapOffset(localRec[index].first,
				chunk_setting.ChunkSize, chunk_setting.MapSize, chunk_setting.MapOffset));
			//local chunk coordinate
			const uvec2 local_coord = STPChunk::calcLocalChunkCoordinate(index, chunk_setting.RenderDistance);

			return STPLocalChunkInformation { local_coord.x, local_coord.y, offset.x, offset.y };
		};
		this->SplatmapInfoCache.clear();

		//put all mandated chunks that should have splatmap computed first
		transform(this->SplatmapComputeChunk.cbegin(), this->SplatmapComputeChunk.cend(),
			back_inserter(this->SplatmapInfoCache), createLocalChunkInfo);
		//then take force regenerate into account
		if (force_regenerate) {
			for (const auto force_entry : *force_regenerate) {
				//need to make sure this entry is not a duplicate in the previous container
				if (const unsigned int index = this->BackBuffer->toLocalIndex(this->BackBuffer->calcLocalIndexCoordinate(force_entry->first));
					this->SplatmapComputeChunk.find(index) == this->SplatmapComputeChunk.cend()) {
					//not a duplicate, insert
					this->SplatmapInfoCache.emplace_back(createLocalChunkInfo(index));
				}
			}
		}
		return this->SplatmapInfoCache;
	}

	/**
	 * @brief Generate a list of chunks that need to be requested from the generator.
	 * @return The chunk request payload.
	*/
	const STPGeneratorManager::STPChunkRequestPayload& generateChunkRequestPayload() {
		this->GeneratorRequestPayload.clear();
		for (const auto [chunk_coord, loaded] : this->BackBuffer->LocalChunkRecord) {
			if (!loaded) {
				//put the local chunk to the request payload if it is not currently loaded in the rendering memory
				this->GeneratorRequestPayload.emplace_back(chunk_coord);
			}
		}
		return this->GeneratorRequestPayload;
	}

	/**
	 * @brief Find the intersection between a given set of chunk, and the chunks in the current render distance.
	 * This function allows filtering out chunks that are not render-able.
	 * @param chunk_set The set of chunks as input.
	 * @param intersection_cache The cache input for storing intermediate data.
	 * @return The intersection result, which is a subset of the input chunk set.
	*/
	const STPGeneratorManager::STPChunkRequestResponseEntry& intersectRenderDistance(
		const STPGeneratorManager::STPChunkRequestResponseEntry& chunk_set, STPRenderDistanceIntersectionCache& intersection_cache) {
		intersection_cache.clear();
		auto& [valid_chunk] = intersection_cache;

		//loop through the chunk set
		//test if this chunk coordinate is in the render distance
		//if it is in the render distance, put that into the result!
		std::copy_if(chunk_set.cbegin(), chunk_set.cend(), back_inserter(valid_chunk),
			[&bb = as_const(*this->BackBuffer)](const auto set_element) { return bb.isLocalChunk(bb.calcLocalIndexCoordinate(set_element->first)); });
		return valid_chunk;
	}

};

STPWorldPipeline::STPWorldPipeline(const STPPipelineSetup& setup) :
	ChunkSetting(*setup.ChunkSetting), BufferStream(STPSmartDeviceObject::makeStream(cudaStreamNonBlocking)),
	Generator(make_unique<STPGeneratorManager>(setup, *this)), Memory(make_unique<STPMemoryManager>(*this)),
	LastCentreLocation(wpMinChunkCoordinate), PipelineWorker(1u) {
	this->ChunkSetting.validate();
}

STPWorldPipeline::~STPWorldPipeline() {
	//sync the chunk loading and make sure the chunk loader has finished before return
	if (this->MapLoader.valid()) {
		//wait for finish first
		this->MapLoader.get();
	}
	this->Memory->unmapBackBuffer();

	//wait for the stream to finish everything
	STP_CHECK_CUDA(cudaStreamSynchronize(this->BufferStream.get()));
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

STPWorldPipeline::STPWorldLoadStatus STPWorldPipeline::load(const dvec3& viewPos) {
	const STPEnvironment::STPChunkSetting& chunk_setting = this->ChunkSetting;
	
	/* -------------------------------- Status Check --------------------------------- */
	const STPChunkLoaderStatus loaderStatus = this->isLoaderBusy();

	if (loaderStatus == STPChunkLoaderStatus::Busy) {
		//loader is working on the back buffer for a previous task
		//can't do anything, ignore the load request
		return STPWorldLoadStatus::BackBufferBusy;
	}
	//we know the loader is not busy
	//check if the central position has changed or not
	if (const ivec2 thisCentrePos = STPChunk::calcWorldChunkCoordinate(
			viewPos - chunk_setting.ChunkOffset, chunk_setting.ChunkSize, chunk_setting.ChunkScale);
		thisCentrePos != this->LastCentreLocation) {
		//centre position changed
		//discard previous unfinished task and move on to the new task
		
		//recalculate loading chunks
		this->Memory->recomputeLocalChunkTable(thisCentrePos);
		this->LastCentreLocation = thisCentrePos;

		//start loading the new rendered chunks
		this->Memory->mapBackBuffer();
	} else {
		//centre position has not changed
		//our current algorithm guarantees all chunks should be loaded when loader finishes
		if (loaderStatus == STPChunkLoaderStatus::Yield) {
			//current task just done completely
			//synchronise the buffer before passing to the user
			this->Memory->unmapBackBuffer();
			this->Memory->swapBuffer();
			return STPWorldLoadStatus::Swapped;
		}
		return STPWorldLoadStatus::Unchanged;
	}

	/* ----------------------------- Asynchronous Chunk Loading ------------------------------- */
	const auto asyncChunkLoader = [&mem_mgr = *this->Memory, &gen_mgr = *this->Generator,
			stream_main = this->BufferStream.get()](const auto& map_data) -> void {
		STPConcurrentStreamManager& stream_mgr = mem_mgr.Worker;
		//given world coordinate...
		const auto toLocalIndex = [&back_buffer = as_const(*mem_mgr.BackBuffer)](const ivec2 coord) noexcept -> unsigned int {
			return back_buffer.toLocalIndex(back_buffer.calcLocalIndexCoordinate(coord));
		};

		//request chunk based on the chunk loading status
		const STPGeneratorManager::STPChunkRequestPayload& requestPayload = mem_mgr.generateChunkRequestPayload();
		const auto& [request_response, require_heightmap_reload] = gen_mgr.requestChunk(requestPayload);
		//find the valid subset for reloading chunk texture
		//this response is different, there might be chunks that are outside the current render area
		//need to filter out
		const STPGeneratorManager::STPChunkRequestResponseEntry* const valid_heightmap_reload = require_heightmap_reload
			? &mem_mgr.intersectRenderDistance(*require_heightmap_reload, mem_mgr.HeightmapIntersectionCache) : nullptr;

		//start auxiliary workers
		stream_mgr.workersWaitMain(stream_main);
		//first, send biomemap from the request to rendering memory
		for (const auto entry : request_response) {
			const auto& [chunk_world_coord, chunk] = *entry;
			mem_mgr.sendChunkToBuffer(map_data, chunk.biomemap(), STPTerrainMapType::Biomemap,
				toLocalIndex(chunk_world_coord), stream_mgr.nextWorker());
		}
		//then, send heightmap based on the response that tells us to update our heightmap buffer
		//If response tells us to reload heightmap, this array is a superset of the main request response,
		//and we don't need to worry about heightmap from the main response.
		//Otherwise we loop through the main response.
		for (const auto entry : valid_heightmap_reload ? *valid_heightmap_reload : request_response) {
			const auto& [chunk_world_coord, chunk] = *entry;
			mem_mgr.sendChunkToBuffer(map_data, chunk.heightmapLow(), STPTerrainMapType::Heightmap,
				toLocalIndex(chunk_world_coord), stream_mgr.nextWorker());
		}
		//end of auxiliary workers
		stream_mgr.mainWaitsWorkers(stream_main);

		//finally, (re)generate splatmap for those chunks that have heightmap (re)loaded
		gen_mgr.generateSplatmap(map_data, mem_mgr.generateSplatmapGeneratorInfo(valid_heightmap_reload));

		//all chunks now should have loaded into the memory, update their statuses
		for_each(requestPayload.cbegin(), requestPayload.cend(),
			[&lr = mem_mgr.BackBuffer->LocalChunkRecord, &toLocalIndex](const auto chunk_coord) { lr[toLocalIndex(chunk_coord)].second = true; });

		//end of chunk loader task
		STP_CHECK_CUDA(cudaStreamSynchronize(stream_main));
	};

	/* ----------------------------- Front Buffer Backup -------------------------------- */
	const auto& backBuffer_ptr = this->Memory->getMappedBackBuffer();
	//Search if any chunk in the front buffer can be reused.
	//If so, copy to the back buffer; if not, clear the chunk.
	this->Memory->reuseBuffer(backBuffer_ptr);

	//group mapped data together and start loading chunk
	this->MapLoader = this->PipelineWorker.enqueue(asyncChunkLoader, std::cref(backBuffer_ptr));
	return STPWorldLoadStatus::BackBufferBusy;
}

const ivec2& STPWorldPipeline::centre() const noexcept {
	return this->LastCentreLocation;
}

STPOpenGL::STPuint STPWorldPipeline::operator[](const STPTerrainMapType type) const noexcept {
	return this->Memory->getMap(type);
}

const STPDiversity::STPTextureFactory& STPWorldPipeline::splatmapGenerator() const noexcept {
	return this->Generator->SplatmapGenerator;
}