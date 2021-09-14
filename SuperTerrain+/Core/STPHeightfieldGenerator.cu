#pragma once
#include <SuperTerrain+/GPGPU/STPHeightfieldGenerator.cuh>

//Simulator
#include <SuperTerrain+/GPGPU/FreeSlip/STPFreeSlipManager.cuh>
#include <SuperTerrain+/GPGPU/STPRainDrop.cuh>

#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Utility/Exception/STPInvalidEnvironment.h>

#include <type_traits>
#include <memory>
//CUDA Device Parameters
#include <device_launch_parameters.h>

//Template definition for the smart device memory
#include <SuperTerrain+/Utility/STPSmartDeviceMemory.tpp>

using namespace SuperTerrainPlus::STPCompute;

using std::vector;
using std::mutex;
using std::unique_lock;
using std::unique_ptr;
using std::optional;
using std::move;
using std::make_unique;
using std::current_exception;
using std::rethrow_exception;

//GLM
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

using glm::ivec2;
using glm::uvec2;
using glm::vec2;
using glm::vec3;

enum class STPHeightfieldGenerator::STPEdgeArrangement : unsigned char {
	TOP_LEFT_CORNER = 0x00u,
	TOP_RIGHT_CORNER = 0x01u,
	BOTTOM_LEFT_CORNER = 0x02u,
	BOTTOM_RIGHT_CORNER = 0x03u,
	TOP_HORIZONTAL_STRIP = 0x10u,
	BOTTOM_HORIZONTAL_STRIP = 0x11u,
	LEFT_VERTICAL_STRIP = 0x12u,
	RIGHT_VERTICAL_STRIP = 0x13u,
	NOT_AN_EDGE = 0xffu
};

/**
 * @brief Init the curand generator for each thread
 * @param rng The random number generator array, it must have the same number of element as thread. e.g.,
 * generating x random number each in 1024 thread needs 1024 rng, each thread will use the same sequence.
 * @param seed The seed for each generator
 * @param raindrop_count The expected number of raindrop, so does the total number of RNG to init
*/
__global__ void curandInitKERNEL(STPHeightfieldGenerator::curandRNG*, unsigned long long, unsigned int);

/**
 * @brief Performing hydraulic erosion for the given heightmap terrain using CUDA parallel computing
 * @param height_storage The floating point heightmap with global-local convertion management
 * @param heightfield_settings - The settings to use to generate heightmap
 * @param rng The random number generator map sequence, independent for each rain drop
*/
__global__ void performErosionKERNEL(STPFreeSlipFloatManager, const SuperTerrainPlus::STPEnvironment::STPHeightfieldSetting*, STPHeightfieldGenerator::curandRNG*);

/**
 * @brief Generate the normal map for the height map within kernel, and combine two maps into a rendering buffer
 * @param heightmap - contains the floating point height map that will be used to generate the normalmap, with free-slip manager
 * @param strength - The strenghth of the generated normal map
 * @param heightfield - will be used to store the output of the normal map in RGB channel, heightmap will be copied to A channel
*/
__global__ void generateRenderingBufferKERNEL(STPFreeSlipFloatManager, float, unsigned short*);


__host__ STPHeightfieldGenerator::STPHeightfieldGenerator(const STPEnvironment::STPChunkSetting& chunk_settings, const STPEnvironment::STPHeightfieldSetting& heightfield_settings,
	const STPDiversityGenerator& diversity_generator, unsigned int hint_level_of_concurrency)
	: generateHeightmap(diversity_generator), Heightfield_Setting_h(heightfield_settings), 
	FreeSlipTable(chunk_settings.FreeSlipChunk, chunk_settings.MapSize) {
	if (!chunk_settings.validate()) {
		throw STPException::STPInvalidEnvironment("Values from STPChunkSetting are not validated");
	}
	if (!heightfield_settings.validate()) {
		throw STPException::STPInvalidEnvironment("Values from STPHeightfieldSetting are not validated");
	}

	const unsigned int num_pixel = this->FreeSlipTable.getDimension().x * this->FreeSlipTable.getDimension().y,
		num_freeslip_pixel = this->FreeSlipTable.getFreeSlipRange().x * this->FreeSlipTable.getFreeSlipRange().y;
	this->TextureBufferAttr.TexturePixel = num_pixel;

	//allocating space
	//heightfield settings
	this->Heightfield_Setting_d = STPSmartDeviceMemory::makeDevice<STPEnvironment::STPHeightfieldSetting>();
	STPcudaCheckErr(cudaMemcpy(this->Heightfield_Setting_d.get(), &this->Heightfield_Setting_h, sizeof(STPEnvironment::STPHeightfieldSetting), cudaMemcpyHostToDevice));
	
	//create memory pool
	cudaMemPoolProps pool_props = { };
	pool_props.allocType = cudaMemAllocationTypePinned;
	pool_props.location.id = 0;
	pool_props.location.type = cudaMemLocationTypeDevice;
	pool_props.handleTypes = cudaMemHandleTypeNone;
	STPcudaCheckErr(cudaMemPoolCreate(&this->MapCacheDevice, &pool_props));
	//TODO: smartly determine the average memory pool size
	cuuint64_t release_thres = (sizeof(float) + sizeof(unsigned short) * 4u) * num_freeslip_pixel * hint_level_of_concurrency;
	STPcudaCheckErr(cudaMemPoolSetAttribute(this->MapCacheDevice, cudaMemPoolAttrReleaseThreshold, &release_thres));
	this->TextureBufferAttr.DeviceMemPool = this->MapCacheDevice;

	//init erosion
	this->setErosionIterationCUDA();
	//init edge table
	this->initEdgeArrangementTable();
}

__host__ STPHeightfieldGenerator::~STPHeightfieldGenerator() {
	STPcudaCheckErr(cudaMemPoolDestroy(this->MapCacheDevice));
	//device ptrs are deleted with custom deleter
}

__host__ void STPHeightfieldGenerator::operator()(STPMapStorage& args, STPGeneratorOperation operation) const {
	//check the availiability of the engine
	if (this->RNG_Map == nullptr) {
		return;
	}
	if (operation == 0u) {
		//no operation is specified, nothing can be done
		return;
	}

	int Mingridsize, gridsize, blocksize;
	//Retrieve all flags
	auto isFlagged = []__host__(STPGeneratorOperation op, STPGeneratorOperation flag) -> bool {
		return (op & flag) != 0u;
	};
	const bool flag[3] = {
		isFlagged(operation, STPHeightfieldGenerator::HeightmapGeneration),
		isFlagged(operation, STPHeightfieldGenerator::Erosion),
		isFlagged(operation, STPHeightfieldGenerator::RenderingBufferGeneration)
	};
	STPcudaCheckErr(cudaSetDevice(0));

	//creating stream so cpu thread can calculate all chunks altogether
	//if exception is thrown during exception, stream will be the last object to be deleted, automatically
	optional<STPSmartStream> stream_buffer;
	cudaStream_t stream;
	//limit the scope for std::optional to control the destructor call
	{
		//heightmap
		optional<STPFreeSlipFloatTextureBuffer> heightmap_buffer;
		optional<STPFreeSlipRenderTextureBuffer> heightfield_buffer;
		optional<STPFreeSlipGenerator::STPFreeSlipFloatManagerAdaptor> heightmap_adaptor;
		//biomemap
		optional<STPFreeSlipSampleTextureBuffer> biomemap_buffer;
		optional<STPFreeSlipGenerator::STPFreeSlipSampleManagerAdaptor> biomemap_adaptor;

		//setup phase
		//we want the stream to not be blocked by default stream
		{
			unique_lock<mutex> stream_lock(this->StreamPool_lock);
			if (this->StreamPool.empty()) {
				//create a new stream
				stream_buffer.emplace(cudaStreamNonBlocking);
			}
			else {
				//grab an exisiting stream
				stream_buffer.emplace(move(this->StreamPool.front()));
				this->StreamPool.pop();
			}
		}
		stream = *stream_buffer;

		//Flag: HeightmapGeneration
		if (flag[0]) {
			//generate a new heightmap using diversity generator and store it to the output later
			//copy biome map to device, and allocate heightmap
			STPFreeSlipFloatTextureBuffer::STPFreeSlipTextureData heightmap_data{ 1u, STPFreeSlipFloatTextureBuffer::STPFreeSlipTextureData::STPMemoryMode::WriteOnly, stream };
			STPFreeSlipSampleTextureBuffer::STPFreeSlipTextureData biomemap_data{ 1u, STPFreeSlipSampleTextureBuffer::STPFreeSlipTextureData::STPMemoryMode::ReadOnly, stream };

			heightmap_buffer.emplace(args.Heightmap32F, heightmap_data, this->TextureBufferAttr);
			biomemap_buffer.emplace(args.Biomemap, biomemap_data, this->TextureBufferAttr);
			biomemap_adaptor.emplace(this->FreeSlipTable(*biomemap_buffer));

			this->generateHeightmap(*heightmap_buffer, *biomemap_adaptor, args.HeightmapOffset, stream);
		}
		else {
			//no generation, use existing
			STPFreeSlipFloatTextureBuffer::STPFreeSlipTextureData heightmap_data{ 1u, STPFreeSlipFloatTextureBuffer::STPFreeSlipTextureData::STPMemoryMode::ReadWrite, stream };
			heightmap_buffer.emplace(args.Heightmap32F, heightmap_data, this->TextureBufferAttr);
		}

		if (flag[1] || flag[2]) {
			//prepare free-slip utility for heightmap
			heightmap_adaptor.emplace(this->FreeSlipTable(*heightmap_buffer));
			STPFreeSlipFloatManager heightmap_slip = (*heightmap_adaptor)(STPFreeSlipLocation::DeviceMemory);

			//Flag: Erosion
			if (flag[1]) {
				const unsigned erosionBrushCache_size = this->Heightfield_Setting_h.getErosionBrushSize() * (sizeof(int) + sizeof(float));
				STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &performErosionKERNEL, erosionBrushCache_size));
				gridsize = (this->Heightfield_Setting_h.RainDropCount + blocksize - 1) / blocksize;

				//erode the heightmap, either from provided heightmap or generated previously
				performErosionKERNEL << <gridsize, blocksize, erosionBrushCache_size, stream >> > (heightmap_slip, this->Heightfield_Setting_d.get(), this->RNG_Map.get());
				STPcudaCheckErr(cudaGetLastError());
			}

			//Flag: RenderingBufferGeneration
			if (flag[2]) {
				//allocate formation memory
				STPFreeSlipRenderTextureBuffer::STPFreeSlipTextureData heightfield_data{ 4u, STPFreeSlipRenderTextureBuffer::STPFreeSlipTextureData::STPMemoryMode::WriteOnly, stream };
				heightfield_buffer.emplace(args.Heightfield16UI, heightfield_data, this->TextureBufferAttr);

				auto det_cacheSize = []__host__ __device__(int blockSize) -> size_t {
					return (blockSize + 2u) * sizeof(float);
				};
				STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&Mingridsize, &blocksize, &generateRenderingBufferKERNEL, det_cacheSize));
				const uvec2 DimblockSize(32u, static_cast<unsigned int>(blocksize) / 32u),
					DimgridSize = (this->FreeSlipTable.getFreeSlipRange() + DimblockSize - 1u) / DimblockSize;

				//get free-slip util and memory
				unsigned short* heightfield_formatted_d = (*heightfield_buffer)(STPFreeSlipLocation::DeviceMemory);
				if (args.Heightfield16UI.size() > 1u) {
					//no need to do copy if freeslip is not enabled
					//this is the way to make sure normalmap is seamless, since the border is already in-sync with other chunks
					this->copyNeighbourEdgeOnly(heightfield_formatted_d, args.Heightfield16UI, this->TextureBufferAttr.TexturePixel, stream);
				}
				//generate normalmap from heightmap and format into rendering buffer
				const uvec2 cacheBlockSize = DimblockSize + 2u;
				const unsigned int cacheSize = cacheBlockSize.x * cacheBlockSize.y * sizeof(float);
				generateRenderingBufferKERNEL << <dim3(DimgridSize.x, DimgridSize.y), dim3(DimblockSize.x, DimblockSize.y), cacheSize, stream >> > (heightmap_slip, this->Heightfield_Setting_h.Strength, heightfield_formatted_d);
				STPcudaCheckErr(cudaGetLastError());
			}
		}

		//Store the result accordingly
		//copy the result back to the host
		//it will call the destructor in texture buffer (optional calls it when goes out of scope), and result will be copied back using CUDA stream
		//this operation is stream ordered
	}

	//waiting for finish before release the stream back to the pool
	STPcudaCheckErr(cudaStreamSynchronize(stream));
	{
		unique_lock<mutex> stream_lock(this->StreamPool_lock);
		this->StreamPool.emplace(move(*stream_buffer));
	}
}

__host__ void STPHeightfieldGenerator::setErosionIterationCUDA() {
	const unsigned int raindrop_count = this->Heightfield_Setting_h.RainDropCount;
	//determine launch parameters
	int Mingridsize, gridsize, blocksize;
	STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &curandInitKERNEL));
	gridsize = (this->Heightfield_Setting_h.RainDropCount + blocksize - 1) / blocksize;

	//make sure all previous takes are finished
	STPcudaCheckErr(cudaDeviceSynchronize());
	//when the raindrop count changes, we need to reallocate and regenerate the rng
	//the the number of rng = the number of the raindrop
	//such that each raindrop has independent rng
	//allocating spaces for rng storage array
	this->RNG_Map = STPSmartDeviceMemory::makeDevice<curandRNG[]>(raindrop_count);
	//and send to kernel
	curandInitKERNEL << <gridsize, blocksize >> > (this->RNG_Map.get(), this->Heightfield_Setting_h.Seed, raindrop_count);
	STPcudaCheckErr(cudaGetLastError());
	STPcudaCheckErr(cudaDeviceSynchronize());
}

__host__ void STPHeightfieldGenerator::copyNeighbourEdgeOnly(unsigned short* device, const vector<unsigned short*>& source, size_t element_count, cudaStream_t stream) const {
	typedef STPHeightfieldGenerator::STPEdgeArrangement STPEA;
	const uvec2& dimension = this->FreeSlipTable.getDimension();
	const unsigned int one_pixel_size = 4u * sizeof(unsigned short);
	const unsigned int pitch = dimension.x * one_pixel_size;
	const unsigned int horizontal_stripe_size = dimension.x * one_pixel_size;
	//we want to cut down the number of copy of column major matrix due to concern about cache
	/**
	* Out copy pattern:				It's more efficient than:
	 ---------------------			+-------------------+
	 |                   |			|                   |
	 |                   |			|                   |
	 |                   |			|                   |
	 |                   |			|                   |
	 ---------------------			+-------------------+
	*/
	//address offset of those situations, eliminate overlap of pixels
	const unsigned int right_vertical_wholerow = (dimension.x - 1u) * 4u,
		left_vertical_skipfirstrow = right_vertical_wholerow + 4u,
		right_vertical_skipfirstrow = left_vertical_skipfirstrow + right_vertical_wholerow,
		bottom_horizontal = dimension.x * 4u * (dimension.y - 1u);

	for (int i = 0; i < source.size(); i++) {
		auto perform_copy = [device, stream, map = source[i], pitch]__host__(size_t start, size_t width_byte, size_t height) -> void {
			STPcudaCheckErr(cudaMemcpy2DAsync(device + start, pitch, map + start, pitch, width_byte, height, cudaMemcpyHostToDevice, stream));
		};

		switch (this->EdgeArrangementTable[i]) {
		case STPEA::TOP_LEFT_CORNER:
			//------------
			//|
			//|
			//|
			//|
			perform_copy(0u, horizontal_stripe_size, 1u);
			perform_copy(left_vertical_skipfirstrow, one_pixel_size, dimension.y - 1u);
			break;
		case STPEA::TOP_RIGHT_CORNER:
			//-------------
			//            |
			//            |
			//            |
			//            |
			perform_copy(0u, horizontal_stripe_size, 1u);
			perform_copy(right_vertical_skipfirstrow, one_pixel_size, dimension.y - 1u);
			break;
		case STPEA::BOTTOM_LEFT_CORNER:
			//|
			//|
			//|
			//|
			//-------------
			perform_copy(bottom_horizontal, horizontal_stripe_size, 1u);
			perform_copy(0u, one_pixel_size, dimension.y - 1u);
			break;
		case STPEA::BOTTOM_RIGHT_CORNER:
			//             |
			//             |
			//             |
			//             |
			//--------------
			perform_copy(bottom_horizontal, horizontal_stripe_size, 1u);
			perform_copy(right_vertical_wholerow, one_pixel_size, dimension.y - 1u);
			break;
		case STPEA::TOP_HORIZONTAL_STRIP:
			//--------------
			//
			//
			//
			//
			perform_copy(0u, horizontal_stripe_size, 1u);
			break;
		case STPEA::BOTTOM_HORIZONTAL_STRIP:
			//
			//
			//
			//
			//--------------
			perform_copy(bottom_horizontal, horizontal_stripe_size, 1u);
			break;
		case STPEA::LEFT_VERTICAL_STRIP:
			//|
			//|
			//|
			//|
			//|
			perform_copy(0u, one_pixel_size, dimension.y);
			break;
		case STPEA::RIGHT_VERTICAL_STRIP:
			//             |
			//             |
			//             |
			//             |
			//             |
			perform_copy(right_vertical_wholerow, one_pixel_size, dimension.y);
			break;
		default:
			//skip every non-edge chunk
			break;
		}
		device += element_count * 4u;
	}
}

__host__ void STPHeightfieldGenerator::initEdgeArrangementTable() {
	typedef STPHeightfieldGenerator::STPEdgeArrangement STPEA;
	const uvec2& freeslip_chunk = this->FreeSlipTable.getFreeSlipChunk();
	const unsigned int num_chunk = freeslip_chunk.x * freeslip_chunk.y;
	if (num_chunk == 1u) {
		//if freeslip logic is not turned on, there's no need to do copy
		//since edge is calculated by neighbour chunks, but without freeslip there's no "other" chunks
		//so the chunk itself needs to compute the border during rendering buffer generation
		return;
	}

	//allocate space
	this->EdgeArrangementTable = make_unique<STPEA[]>(num_chunk);
	for (unsigned int chunkID = 0u; chunkID < num_chunk; chunkID++) {
		STPEA& current_entry = this->EdgeArrangementTable[chunkID];
		const uvec2 chunkCoord(chunkID % freeslip_chunk.x, chunkID / freeslip_chunk.x);

		//some basic boolean logic to determine our "frame"
		if (chunkCoord.x == 0u) {
			if (chunkCoord.y == 0u) {
				current_entry = STPEA::TOP_LEFT_CORNER;
				continue;
			}
			if (chunkCoord.y == freeslip_chunk.y - 1u) {
				current_entry = STPEA::BOTTOM_LEFT_CORNER;
				continue;
			}
			current_entry = STPEA::LEFT_VERTICAL_STRIP;
			continue;
		}
		if (chunkCoord.x == freeslip_chunk.x - 1u) {
			if (chunkCoord.y == 0u) {
				current_entry = STPEA::TOP_RIGHT_CORNER;
				continue;
			}
			if (chunkCoord.y == freeslip_chunk.y - 1u) {
				current_entry = STPEA::BOTTOM_RIGHT_CORNER;
				continue;
			}
			current_entry = STPEA::RIGHT_VERTICAL_STRIP;
			continue;
		}

		if (chunkCoord.y == 0u) {
			current_entry = STPEA::TOP_HORIZONTAL_STRIP;
			continue;
		}
		if (chunkCoord.y == freeslip_chunk.y - 1u) {
			current_entry = STPEA::BOTTOM_HORIZONTAL_STRIP;
			continue;
		}

		//we can safely ignore edge that's not an edge chunk
		current_entry = STPEA::NOT_AN_EDGE;
	}
}

__global__ void curandInitKERNEL(STPHeightfieldGenerator::curandRNG* rng, unsigned long long seed, unsigned int raindrop_count) {
	//current working index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= raindrop_count) {
		return;
	}

	//the same seed but we are looking for different sequence
	curand_init(seed, static_cast<unsigned long long>(index), 0, &rng[index]);
}

__global__ void performErosionKERNEL(STPFreeSlipFloatManager heightmap_storage, const SuperTerrainPlus::STPEnvironment::STPHeightfieldSetting* heightfield_settings, STPHeightfieldGenerator::curandRNG* rng) {
	//current working index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= heightfield_settings->RainDropCount) {
		return;
	}

	//convert to (base, dimension - 1]
	//range: dimension
	//Generate the raindrop at the central chunk only
	__shared__ uvec2 base;
	__shared__ uvec2 range;
	if (threadIdx.x == 0u) {
		const uvec2& dimension = heightmap_storage.Data->Dimension;

		base = dimension - 1u,
		range = (heightmap_storage.Data->FreeSlipChunk / 2u) * dimension;
	}
	__syncthreads();

	//generating random location
	//first we generate the number (0.0f, 1.0f]
	vec2 initPos = vec2(curand_uniform(&rng[index]), curand_uniform(&rng[index]));
	//range convertion
	initPos *= base;
	initPos += range;

	//spawn in the raindrop
	STPRainDrop droplet(initPos, heightfield_settings->initWaterVolume, heightfield_settings->initSpeed);
	droplet.Erode(static_cast<const SuperTerrainPlus::STPEnvironment::STPRainDropSetting*>(heightfield_settings), heightmap_storage);
}

__global__ void generateRenderingBufferKERNEL(STPFreeSlipFloatManager heightmap, float strength, unsigned short* heightfield) {
	//the current working pixel
	const uvec2 block = uvec2(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y),
		local_thread = uvec2(threadIdx.x, threadIdx.y),
		thread = local_thread + block;
	const unsigned int threadperblock = blockDim.x * blockDim.y;
	const uvec2& freeslip_range = heightmap.Data->FreeSlipRange;
	if (thread.x >= freeslip_range.x || thread.y >= freeslip_range.y) {
		return;
	}

	const uvec2& dimension = heightmap.Data->FreeSlipRange;
	auto float2short = []__device__(float input) -> unsigned short {
		return static_cast<unsigned short>(input * 0xFFFFu);
	};

	//Cache heightmap the current thread block needs since each pixel is accessed upto 9 times.
	extern __shared__ float heightmapCache[];
	//each thread needs to access a 3x3 matrix around the current pixel, so we need to take the edge into consideration
	const uvec2 cacheSize = uvec2(blockDim.x, blockDim.y) + 2u;
	unsigned int iteration = 0u;
	const unsigned int cacheSize_total = cacheSize.x * cacheSize.y;

	while (iteration < cacheSize_total) {
		const unsigned int cacheIdx = (threadIdx.x + blockDim.x * threadIdx.y) + iteration;
		const uvec2 worker = block + uvec2(cacheIdx % cacheSize.x, cacheIdx / cacheSize.x);
		//worker index may be zero, and 0u - 1u will become UINT32_MAX, so we should cast it to int and it will be come -1 thus correctly clampped
		const uvec2 clamppeWorkerIdx = static_cast<uvec2>(glm::clamp(static_cast<ivec2>(worker) - 1, ivec2(0), static_cast<ivec2>(dimension - 1u)));
		const unsigned int workerIdx = clamppeWorkerIdx.x + clamppeWorkerIdx.y * dimension.x;

		if (cacheIdx < cacheSize_total) {
			//make sure index don't get out of bound
			//start caching from (x-1, y-1) until (x+1, y+1)
			heightmapCache[cacheIdx] = heightmap[workerIdx];
		}
		//warp around to reuse some threads to finish all compute
		iteration += threadperblock;
	}
	__syncthreads();

	if ((heightmap.Data->FreeSlipChunk.x * heightmap.Data->FreeSlipChunk.y) > 1 && 
		(thread.x == 0 || thread.y == 0 || thread.x == freeslip_range.x - 1 || thread.y == freeslip_range.y - 1)) {
		//if freeslip is not turned on, we need to calculate the edge pixel for this chunk
		//otherwise, do not touch the border pixel since border pixel is calculated seamlessly by other chunks
		return;
	}

	//load the cells from heightmap, remember the height map only contains one color channel
	//using Sobel fitering
	//Cache index
	const uvec2 cache = local_thread + 1u;
	//no need to pass by reference since vec2 type a big as a pointer
	auto loadCache = [cacheIdx = cache, horizontalCacheSize = cacheSize.x]__device__(float* cache, ivec2 offset) -> float {
		const uvec2 coord = static_cast<uvec2>(static_cast<ivec2>(cacheIdx) + offset);
		return cache[coord.x + coord.y * horizontalCacheSize];
	};

	float cell[8];
	cell[0] = loadCache(heightmapCache, ivec2(-1, -1));
	cell[1] = loadCache(heightmapCache, ivec2(0, -1));
	cell[2] = loadCache(heightmapCache, ivec2(+1, -1));
	cell[3] = loadCache(heightmapCache, ivec2(-1, 0));
	cell[4] = loadCache(heightmapCache, ivec2(+1, 0));
	cell[5] = loadCache(heightmapCache, ivec2(-1, +1));
	cell[6] = loadCache(heightmapCache, ivec2(0, +1));
	cell[7] = loadCache(heightmapCache, ivec2(+1, +1));
	//apply the filtering kernel matrix
	vec3 normal;
	normal.z = 1.0f / strength;
	normal.x = cell[0] + 2 * cell[3] + cell[5] - (cell[2] + 2 * cell[4] + cell[7]);
	normal.y = cell[0] + 2 * cell[1] + cell[2] - (cell[5] + 2 * cell[6] + cell[7]);
	//normalize
	normal = glm::normalize(normal);
	//clamp to [0,1], was [-1,1]
	normal = (glm::clamp(normal, -1.0f, 1.0f) + 1.0f) / 2.0f;
	
	//copy to the output, RGBA32F
	const unsigned int index = heightmap(thread.x + thread.y * dimension.x);
	heightfield[index * 4] = float2short(normal.x);//R
	heightfield[index * 4 + 1] = float2short(normal.y);//G
	heightfield[index * 4 + 2] = float2short(normal.z);//B
	heightfield[index * 4 + 3] = float2short(heightmapCache[cache.x + cache.y * cacheSize.x]);//A
}