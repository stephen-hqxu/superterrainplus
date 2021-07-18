#pragma once
#include "STPHeightfieldGenerator.cuh"

#define STP_EXCEPTION_ON_ERROR
#include "STPDeviceErrorHandler.cuh"

#include <memory>

using namespace SuperTerrainPlus::STPCompute;

using std::vector;
using std::mutex;
using std::unique_lock;
using std::unique_ptr;
using std::make_unique;

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
 * @brief Find the unit vector of the input vector
 * @param vec3 - Vector input
 * @return Unit vector of the input
*/
__device__ __inline__ float3 normalize3DKERNEL(float3);

/**
 * @brief Performing inverse linear interpolation for each value on the heightmap to scale it within [0,1] using CUDA kernel
 * @param minVal The mininmum value that can apperar in this height map
 * @param maxVal The maximum value that can apperar in this height map
 * @param value The input value
 * @return The interpolated value
*/
__device__ __inline__ float InvlerpKERNEL(float, float, float);

/**
 * @brief Init the curand generator for each thread
 * @param rng The random number generator array, it must have the same number of element as thread. e.g.,
 * generating x random number each in 1024 thread needs 1024 rng, each thread will use the same sequence.
 * @param seed The seed for each generator
 * @param raindrop_count The expected number of raindrop, so does the total number of RNG to init
*/
__global__ void curandInitKERNEL(STPHeightfieldGenerator::curandRNG*, unsigned long long, unsigned int);

/**
 * @brief Generate our epic height map using simplex noise function within the CUDA kernel
 * @param noise_fun - The heightfield generator that's going to use
 * @param heightfield_settings - The settings to use to generate heightmap
 * @param height_storage - The pointer to a location where the heightmap will be stored
 * @param dimension - The width and height of the generated heightmap
 * @param half_dimension - Precomputed dimension/2 so the kernel don't need to repeatly compute that
 * @param offset - Controlling the offset on x, y directions
*/
__global__ void generateHeightmapKERNEL(const STPSimplexNoise*, const SuperTerrainPlus::STPSettings::STPHeightfieldSettings*, float*, uint2, float2, float2);

/**
 * @brief Performing hydraulic erosion for the given heightmap terrain using CUDA parallel computing
 * @param height_storage The heightmap with global-local convertion management
 * @param heightfield_settings - The settings to use to generate heightmap
 * @param rng The random number generator map sequence, independent for each rain drop
*/
__global__ void performErosionKERNEL(STPRainDrop::STPFreeSlipManager, const SuperTerrainPlus::STPSettings::STPHeightfieldSettings*, STPHeightfieldGenerator::curandRNG*);

/**
 * @brief Generate the normal map for the height map within kernel, and combine two maps into a rendering buffer
 * @param heightmap - contains the height map that will be used to generate the normalmap, with free-slip manager
 * @param strength - The strenghth of the generated normal map
 * @param heightfield - will be used to store the output of the normal map in RGB channel, heightmap will be copied to A channel
*/
__global__ void generateRenderingBufferKERNEL(STPRainDrop::STPFreeSlipManager, float, unsigned short*);

/**
 * @brief Generate a new global to local index table
 * @param output The generated table. Should be preallocated with size sizeof(unsigned int) * chunkRange.x * mapSize.x * chunkRange.y * mapSize.y
 * @param rowCount The number of row in the global index table, which is equivalent to chunkRange.x * mapSize.x
 * @param chunkRange The number of chunk (or locals)
 * @param tableSize The x,y dimension of the table
 * @param mapSize The dimension of the map
*/
__global__ void initGlobalLocalIndexKERNEL(unsigned int*, unsigned int, uint2, uint2, uint2);

/*
 * @brief Copy block of memory from device to host and split into chunks.
 * Used by function operator()
 * @param dest - The destination chunks
 * @param host - Host pinned memory acts as buffer
 * @param device - Device source memory
 * @param block_size - The total size of the host and device memory, in byte.
 * @param individual_size - The size of one chunk, in byte.
 * @param element_count - The number of pixel in one chunk
 * @param stream - Async CUDA stream
*/
template<typename T>
__host__ void blockcopy_d2h(vector<T*>& dest, T* host, T* device, size_t block_size, size_t individual_size, size_t element_count, cudaStream_t stream) {
	STPcudaCheckErr(cudaMemcpyAsync(host, device, block_size, cudaMemcpyDeviceToHost, stream));
	for (T* map : dest) {
		STPcudaCheckErr(cudaMemcpyAsync(map, host, individual_size, cudaMemcpyHostToHost, stream));
		host += element_count;
	}
}

/*
 * @brief Copy block of memory from host, split chunks into maps and then copy to device
 * Used by function operator()
 * @param host - Host pinned memory acts as buffer
 * @param device - Device destination memory
 * @param souce - The source chunks
 * @param block_size - The total size of the host and device memory, in byte.
 * @param individual_size - The size of one chunk, in byte.
 * @param element_count - The number of pixel in one chunk
 * @param stream - Async CUDA stream
*/
template<typename T>
__host__ void blockcopy_h2d(T* device, T* host, const vector<T*>& source, size_t block_size, size_t individual_size, size_t element_count, cudaStream_t stream) {
	unsigned int base = 0u;
	for (const T* map : source) {
		STPcudaCheckErr(cudaMemcpyAsync(host + base, map, individual_size, cudaMemcpyHostToHost, stream));
		base += element_count;
	}
	STPcudaCheckErr(cudaMemcpyAsync(device, host, block_size, cudaMemcpyHostToDevice, stream));
}

__host__ void* STPHeightfieldGenerator::STPHeightfieldHostAllocator::allocate(size_t count) {
	void* mem = nullptr;
	STPcudaCheckErr(cudaMallocHost(&mem, count, cudaHostAllocMapped));
	return mem;
}

__host__ void STPHeightfieldGenerator::STPHeightfieldHostAllocator::deallocate(size_t count, void* ptr) {
	STPcudaCheckErr(cudaFreeHost(ptr));
}

__host__ void* STPHeightfieldGenerator::STPHeightfieldNonblockingStreamAllocator::allocate(size_t count) {
	cudaStream_t stream;
	STPcudaCheckErr(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	return reinterpret_cast<void*>(stream);
}

__host__ void STPHeightfieldGenerator::STPHeightfieldNonblockingStreamAllocator::deallocate(size_t count, void* stream) {
	STPcudaCheckErr(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
}

template<typename T>
void STPHeightfieldGenerator::STPDeviceDeleter<T>::operator()(T* ptr) const {
	STPcudaCheckErr(cudaFree(ptr));
}

__host__ STPHeightfieldGenerator::STPHeightfieldGenerator(const STPSettings::STPSimplexNoiseSettings& noise_settings, 
	const STPSettings::STPChunkSettings& chunk_settings, const STPSettings::STPHeightfieldSettings& heightfield_settings, 
	STPDiversity::STPBiomeFactory& factory, unsigned int hint_level_of_concurrency)
	: simplex_h(&noise_settings), Noise_Settings(noise_settings), FreeSlipChunk(make_uint2(chunk_settings.FreeSlipChunk.x, chunk_settings.FreeSlipChunk.y)), 
	Heightfield_Settings_h(heightfield_settings), biome(factory) {
	const unsigned int num_pixel = chunk_settings.MapSize.x * chunk_settings.MapSize.y,
		num_freeslip_chunk = chunk_settings.FreeSlipChunk.x * chunk_settings.FreeSlipChunk.y;

	//allocating space
	//simplex noise generator
	STPSimplexNoise* simplex_cache;
	STPcudaCheckErr(cudaMalloc(&simplex_cache, sizeof(STPSimplexNoise)));
	STPcudaCheckErr(cudaMemcpy(simplex_cache, &this->simplex_h, sizeof(STPSimplexNoise), cudaMemcpyHostToDevice));
	this->simplex_d = unique_ptr_d<STPSimplexNoise>(simplex_cache);
	//heightfield settings
	STPSettings::STPHeightfieldSettings* hfs_cache;
	STPcudaCheckErr(cudaMalloc(&hfs_cache, sizeof(STPSettings::STPHeightfieldSettings)));
	STPcudaCheckErr(cudaMemcpy(hfs_cache, &this->Heightfield_Settings_h, sizeof(STPSettings::STPHeightfieldSettings), cudaMemcpyHostToDevice));
	this->Heightfield_Settings_d = unique_ptr_d<STPSettings::STPHeightfieldSettings>(hfs_cache);
	
	//create memory pool
	cudaMemPoolProps pool_props = { };
	pool_props.allocType = cudaMemAllocationTypePinned;
	pool_props.location.id = 0;
	pool_props.location.type = cudaMemLocationTypeDevice;
	pool_props.handleTypes = cudaMemHandleTypeNone;
	STPcudaCheckErr(cudaMemPoolCreate(&this->MapCacheDevice, &pool_props));
	//TODO: smartly determine the average memory pool size
	cuuint64_t release_thres = (sizeof(float) + sizeof(unsigned short) * 4u) * num_freeslip_chunk * num_pixel * hint_level_of_concurrency;
	STPcudaCheckErr(cudaMemPoolSetAttribute(this->MapCacheDevice, cudaMemPoolAttrReleaseThreshold, &release_thres));

	//init erosion
	this->setErosionIterationCUDA();
	//set global local index
	this->initLocalGlobalIndexCUDA();
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
	//check the availability of biome stuff
	/*if (!this->BiomeDictionary_d) {
		return;
	}*/
	if (operation == 0u) {
		//no operation is specified, nothing can be done
		return;
	}

	std::exception_ptr exp;
	int Mingridsize, gridsize, blocksize;
	dim3 Dimgridsize, Dimblocksize;
	//allocating spaces for texture, storing on device
	//this is the size for a texture in one channel
	const unsigned int num_pixel = this->Noise_Settings.Dimension.x * this->Noise_Settings.Dimension.y;
	const unsigned int map_size = num_pixel * sizeof(float);
	const unsigned int map16ui_size = num_pixel * sizeof(unsigned short) * 4u;
	//if free-slip erosion is disabled, it should be one.
	//if enabled, it should be the product of two dimension in free-slip chunk.
	const unsigned int freeslip_chunk_total = args.Heightmap32F.size();
	const uint2 freeslip_dimension = make_uint2(this->Noise_Settings.Dimension.x * this->FreeSlipChunk.x, this->Noise_Settings.Dimension.y * this->FreeSlipChunk.y);
	const unsigned int freeslip_pixel = freeslip_chunk_total * num_pixel;
	const unsigned int map_freeslip_size = freeslip_pixel * sizeof(float);
	const unsigned int map16ui_freeslip_size = freeslip_pixel * sizeof(unsigned short) * 4u;
	//heightmap
	float* heightfield_freeslip_d = nullptr, *heightfield_freeslip_h = nullptr;
	unsigned short* heightfield_formatted_d = nullptr, *heightfield_formatted_h = nullptr;

	//Retrieve all flags
	auto isFlagged = []__host__(STPGeneratorOperation op, STPGeneratorOperation flag) -> bool {
		return (op & flag) != 0u;
	};
	const bool flag[4] = {
		isFlagged(operation, STPHeightfieldGenerator::BiomemapGeneration),
		isFlagged(operation, STPHeightfieldGenerator::HeightmapGeneration),
		isFlagged(operation, STPHeightfieldGenerator::Erosion),
		isFlagged(operation, STPHeightfieldGenerator::RenderingBufferGeneration)
	};

	STPcudaCheckErr(cudaSetDevice(0));
	//setup phase
	//creating stream so cpu thread can calculate all chunks altogether
	cudaStream_t stream = nullptr;
	//we want the stream to not be blocked by default stream
	{
		unique_lock<mutex> stream_lock(this->StreamPool_lock);
		stream = reinterpret_cast<cudaStream_t>(this->StreamPool.allocate(1ull));
	}
	//memory allocation
	//Device
	//FP32
	//we need heightmap for computation regardlessly
	STPcudaCheckErr(cudaMallocFromPoolAsync(&heightfield_freeslip_d, map_freeslip_size, this->MapCacheDevice, stream));
	//INT16
	if (flag[3]) {
		STPcudaCheckErr(cudaMallocFromPoolAsync(&heightfield_formatted_d, map16ui_freeslip_size, this->MapCacheDevice, stream));
	}
	//Host
	{
		unique_lock<mutex> lock(this->MapCachePinned_lock);
		//FP32
		heightfield_freeslip_h = reinterpret_cast<float*>(this->MapCachePinned.allocate(map_freeslip_size));
		//INT16
		if (flag[3]) {
			heightfield_formatted_h = reinterpret_cast<unsigned short*>(this->MapCachePinned.allocate(map16ui_freeslip_size));
		}
	}

	//Flag: BiomemapGeneration
	if (flag[0]) {
		auto& offset = args.HeightmapOffset;
		//generate biomemap on host, we always generate one map
		//since biomemap is discrete, we need to round the pixel
		this->biome(args.Biomemap.front(), glm::ivec3(static_cast<int>(roundf(offset.x)), 1, static_cast<int>(roundf(offset.y))));
	}
	
	//Flag: HeightmapGeneration
	if (flag[1]) {
		STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &generateHeightmapKERNEL));
		Dimblocksize = dim3(32, blocksize / 32);
		Dimgridsize = dim3((this->Noise_Settings.Dimension.x + Dimblocksize.x - 1) / Dimblocksize.x, (this->Noise_Settings.Dimension.y + Dimblocksize.y - 1) / Dimblocksize.y);

		//generate a new heightmap and store it to the output later
		generateHeightmapKERNEL << <Dimgridsize, Dimblocksize, 0, stream >> > (this->simplex_d.get(), this->Heightfield_Settings_d.get(), heightfield_freeslip_d,
			this->Noise_Settings.Dimension, make_float2(1.0f * this->Noise_Settings.Dimension.x / 2.0f, 1.0f * this->Noise_Settings.Dimension.y / 2.0f), args.HeightmapOffset);
		STPcudaCheckErr(cudaGetLastError());
	}
	else {
		//no generation, use existing
		try {
			blockcopy_h2d(heightfield_freeslip_d, heightfield_freeslip_h, args.Heightmap32F, map_freeslip_size, map_size, num_pixel, stream);
		}
		catch (...) {
			exp = std::current_exception();
		}
		
	}

	if (flag[2] || flag[3]) {
		const STPRainDrop::STPFreeSlipManager heightmap_slip(heightfield_freeslip_d, this->GlobalLocalIndex.get(), this->FreeSlipChunk, this->Noise_Settings.Dimension);

		//Flag: Erosion
		if (flag[2]) {
			const unsigned erosionBrushCache_size = this->Heightfield_Settings_h.getErosionBrushSize() * (sizeof(int) + sizeof(float));
			STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &performErosionKERNEL, erosionBrushCache_size));
			gridsize = (this->Heightfield_Settings_h.RainDropCount + blocksize - 1) / blocksize;

			//erode the heightmap, either from provided heightmap or generated previously
			performErosionKERNEL << <gridsize, blocksize, erosionBrushCache_size, stream >> > (heightmap_slip, this->Heightfield_Settings_d.get(), this->RNG_Map.get());
			STPcudaCheckErr(cudaGetLastError());
		}

		//Flag: RenderingBufferGeneration
		if (flag[3]) {
			auto det_cacheSize = []__host__ __device__(int blockSize) -> size_t {
				return (blockSize + 2u) * sizeof(float);
			};
			STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&Mingridsize, &blocksize, &generateRenderingBufferKERNEL, det_cacheSize));
			Dimblocksize = dim3(32, blocksize / 32);
			Dimgridsize = dim3((freeslip_dimension.x + Dimblocksize.x - 1) / Dimblocksize.x, (freeslip_dimension.y + Dimblocksize.y - 1) / Dimblocksize.y);

			if (freeslip_chunk_total > 1u) {
				//no need to do copy if freeslip is not enabled
				try {
					//this is the way to make sure normalmap is seamless, since the border is already in-sync with other chunks
					this->copyNeighbourEdgeOnly(heightfield_formatted_d, args.Heightfield16UI, num_pixel, stream);
				}
				catch (...) {
					exp = std::current_exception();
				}
			}
			//generate normalmap from heightmap and format into rendering buffer
			const unsigned int cacheSize = (Dimblocksize.x + 2u) * (Dimblocksize.y + 2u) * sizeof(float);
			generateRenderingBufferKERNEL << <Dimgridsize, Dimblocksize, cacheSize, stream >> > (heightmap_slip, this->Heightfield_Settings_h.Strength, heightfield_formatted_d);
			STPcudaCheckErr(cudaGetLastError());
		}
	}
	
	//Store the result accordingly
	//copy the result back to the host
	//heightmap will always be available
	try {
		if (flag[1] || flag[2]) {
			//copy all heightmap chunks back if heightmap has been modified
			blockcopy_d2h(args.Heightmap32F, heightfield_freeslip_h, heightfield_freeslip_d, map_freeslip_size, map_size, num_pixel, stream);
		}
		//copy the rendering buffer result if enabled
		if (flag[3]) {
			//copy heightfield
			blockcopy_d2h(args.Heightfield16UI, heightfield_formatted_h, heightfield_formatted_d, map16ui_freeslip_size, map16ui_size, num_pixel * 4u, stream);
		}
	}
	catch (...) {
		exp = std::current_exception();
	}

	//Finish up the rest, clear up when the device is ready
	//nullptr means not allocated
	if (heightfield_freeslip_d != nullptr) {
		STPcudaCheckErr(cudaFreeAsync(heightfield_freeslip_d, stream));
	}
	if (heightfield_formatted_d != nullptr) {
		STPcudaCheckErr(cudaFreeAsync(heightfield_formatted_d, stream));
	}
	//waiting for finish
	STPcudaCheckErr(cudaStreamSynchronize(stream));
	{
		unique_lock<mutex> stream_lock(this->StreamPool_lock);
		this->StreamPool.deallocate(1ull, reinterpret_cast<void*>(stream));
	}

	{
		unique_lock<mutex> lock(this->MapCachePinned_lock);
		if (heightfield_freeslip_h != nullptr) {
			this->MapCachePinned.deallocate(map_freeslip_size, heightfield_freeslip_h);
		}
		if (heightfield_formatted_h != nullptr) {
			this->MapCachePinned.deallocate(map16ui_freeslip_size, heightfield_formatted_h);
		}
	}

	if (exp) {
		std::rethrow_exception(exp);
	}
}

__host__ void STPHeightfieldGenerator::setErosionIterationCUDA() {
	const unsigned int raindrop_count = this->Heightfield_Settings_h.RainDropCount;
	//determine launch parameters
	int Mingridsize, gridsize, blocksize;
	STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &curandInitKERNEL));
	gridsize = (this->Heightfield_Settings_h.RainDropCount + blocksize - 1) / blocksize;

	//make sure all previous takes are finished
	STPcudaCheckErr(cudaDeviceSynchronize());
	//when the raindrop count changes, we need to reallocate and regenerate the rng
	//the the number of rng = the number of the raindrop
	//such that each raindrop has independent rng
	//allocating spaces for rng storage array
	curandRNG* rng_cache;
	STPcudaCheckErr(cudaMalloc(&rng_cache, sizeof(curandRNG) * raindrop_count));
	//and send to kernel
	curandInitKERNEL << <gridsize, blocksize >> > (rng_cache, this->Noise_Settings.Seed, raindrop_count);
	STPcudaCheckErr(cudaGetLastError());
	STPcudaCheckErr(cudaDeviceSynchronize());
	this->RNG_Map = unique_ptr_d<curandRNG>(rng_cache);
}

__host__ void STPHeightfieldGenerator::copyNeighbourEdgeOnly(unsigned short* device, const vector<unsigned short*>& source, size_t element_count, cudaStream_t stream) const {
	typedef STPHeightfieldGenerator::STPEdgeArrangement STPEA;
	const uint2& dimension = this->Noise_Settings.Dimension;
	const unsigned int one_pixel_size = 4u * sizeof(unsigned short);
	const unsigned int pitch = dimension.x * one_pixel_size;
	const unsigned int horizontal_stripe_size = this->Noise_Settings.Dimension.x * one_pixel_size;
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
		auto perform_copy = [device, stream, map = source[i], &pitch]__host__(size_t start, size_t width_byte, size_t height) -> void {
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

__host__ void STPHeightfieldGenerator::initLocalGlobalIndexCUDA() {
	const uint2& dimension = this->Noise_Settings.Dimension;
	const uint2& range = this->FreeSlipChunk;
	const uint2 global_dimension = make_uint2(range.x * dimension.x, range.y * dimension.y);
	//launch parameters
	int Mingridsize, blocksize;
	dim3 Dimgridsize, Dimblocksize;
	STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &initGlobalLocalIndexKERNEL));
	Dimblocksize = dim3(32, blocksize / 32);
	Dimgridsize = dim3((global_dimension.x + Dimblocksize.x - 1) / Dimblocksize.x, (global_dimension.y + Dimblocksize.y - 1) / Dimblocksize.y);

	//Don't generate the table when FreeSlipChunk.xy are both 1, and in STPRainDrop don't use the table
	if (range.x == 1u && range.y == 1u) {
		this->GlobalLocalIndex = nullptr;
		return;
	}

	//make sure all previous takes are finished
	STPcudaCheckErr(cudaDeviceSynchronize());
	//allocation
	unsigned int* gli_cache;
	STPcudaCheckErr(cudaMalloc(&gli_cache, sizeof(unsigned int) * global_dimension.x * global_dimension.y));
	//compute
	initGlobalLocalIndexKERNEL << <Dimgridsize, Dimblocksize >> > (gli_cache, global_dimension.x, range, global_dimension, dimension);
	STPcudaCheckErr(cudaGetLastError());
	STPcudaCheckErr(cudaDeviceSynchronize());
	this->GlobalLocalIndex = unique_ptr_d<unsigned int>(gli_cache);
}

__host__ void STPHeightfieldGenerator::initEdgeArrangementTable() {
	typedef STPHeightfieldGenerator::STPEdgeArrangement STPEA;
	const unsigned int num_chunk = this->FreeSlipChunk.x * this->FreeSlipChunk.y;
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
		const uint2 chunkCoord = make_uint2(chunkID % this->FreeSlipChunk.x, static_cast<unsigned int>(floorf(1.0f * chunkID / this->FreeSlipChunk.x)));

		//some basic boolean logic to determine our "frame"
		if (chunkCoord.x == 0u) {
			if (chunkCoord.y == 0u) {
				current_entry = STPEA::TOP_LEFT_CORNER;
				continue;
			}
			if (chunkCoord.y ==  this->FreeSlipChunk.y - 1u) {
				current_entry = STPEA::BOTTOM_LEFT_CORNER;
				continue;
			}
			current_entry = STPEA::LEFT_VERTICAL_STRIP;
			continue;
		}
		if (chunkCoord.x == this->FreeSlipChunk.x - 1u) {
			if (chunkCoord.y == 0u) {
				current_entry = STPEA::TOP_RIGHT_CORNER;
				continue;
			}
			if (chunkCoord.y == this->FreeSlipChunk.y - 1u) {
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
		if (chunkCoord.y == this->FreeSlipChunk.y - 1u) {
			current_entry = STPEA::BOTTOM_HORIZONTAL_STRIP;
			continue;
		}

		//we can safely ignore edge that's not an edge chunk
		current_entry = STPEA::NOT_AN_EDGE;
	}
}

__device__ __inline__ float3 normalize3DKERNEL(float3 vec3) {
	const float length = sqrtf(powf(vec3.x, 2) + powf(vec3.y, 2) + powf(vec3.z, 2));
	return make_float3(fdividef(vec3.x, length), fdividef(vec3.y, length), fdividef(vec3.z, length));
}

__device__ __inline__ float InvlerpKERNEL(float minVal, float maxVal, float value) {
	//lerp the noiseheight to [0,1]
	return __saturatef(fdividef(value - minVal, maxVal - minVal));
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

__global__ void generateHeightmapKERNEL(const STPSimplexNoise* noise_fun, const SuperTerrainPlus::STPSettings::STPHeightfieldSettings* heightfield_settings, float* height_storage,
	uint2 dimension, float2 half_dimension, float2 offset) {
	//the current working pixel
	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= dimension.x || y >= dimension.y) {
		return;
	}

	float amplitude = 1.0f, frequency = 1.0f, noiseheight = 0.0f;
	float min = 0.0f, max = 0.0f;//The min and max indicates the range of the multi-phased simplex function, not the range of the output texture
	//multiple phases of noise
	for (int i = 0; i < heightfield_settings->Octave; i++) {
		float sampleX = ((1.0 * x - half_dimension.x) + offset.x) / heightfield_settings->Scale * frequency, //subtract the half width and height can make the scaling focus at the center
			sampleY = ((1.0 * y - half_dimension.y) + offset.y) / heightfield_settings->Scale * frequency;//since the y is inverted we want to filp it over
		noiseheight += noise_fun->simplex2D(sampleX, sampleY) * amplitude;

		//calculate the min and max
		min -= 1.0f * amplitude;
		max += 1.0f * amplitude;
		//scale the parameters
		amplitude *= heightfield_settings->Persistence;
		frequency *= heightfield_settings->Lacunarity;
	}
	
	//interpolate and clamp the value within [0,1], was [min,max]
	noiseheight = InvlerpKERNEL(min, max, noiseheight);
	//finally, output the texture
	height_storage[x + y * dimension.x] = noiseheight;//we have only allocated R32F format;
	
	return;
}

__global__ void performErosionKERNEL(STPRainDrop::STPFreeSlipManager heightmap_storage, const SuperTerrainPlus::STPSettings::STPHeightfieldSettings* heightfield_settings, STPHeightfieldGenerator::curandRNG* rng) {
	//current working index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= heightfield_settings->RainDropCount) {
		return;
	}

	//convert to (base, dimension - 1]
	//range: dimension
	//Generate the raindrop at the central chunk only
	__shared__ uint4 area;
	if (threadIdx.x == 0u) {
		const uint2 dimension = heightmap_storage.Dimension;
		const uint2 freeslip_chunk = heightmap_storage.FreeSlipChunk;
		area = make_uint4(
			//base x
			static_cast<unsigned int>(1.0f * dimension.x - 1.0f),
			//range x
			static_cast<unsigned int>(floorf(freeslip_chunk.x / 2.0f) * dimension.x),
			//base z
			static_cast<unsigned int>(1.0f * dimension.y - 1.0f),
			//range z
			static_cast<unsigned int>(floorf(freeslip_chunk.y / 2.0f) * dimension.y)
		);
	}
	__syncthreads();

	//generating random location
	//first we generate the number (0.0f, 1.0f]
	float2 initPos = make_float2(curand_uniform(&rng[index]), curand_uniform(&rng[index]));
	//range convertion
	initPos.x *= area.x;
	initPos.x += area.y;
	initPos.y *= area.z;
	initPos.y += area.w;

	//spawn in the raindrop
	STPRainDrop droplet(initPos, heightfield_settings->initWaterVolume, heightfield_settings->initSpeed);
	droplet.Erode((const SuperTerrainPlus::STPSettings::STPRainDropSettings*)heightfield_settings, heightmap_storage);
}

__global__ void generateRenderingBufferKERNEL(STPRainDrop::STPFreeSlipManager heightmap, float strength, unsigned short* heightfield) {
	//the current working pixel
	const unsigned int x_b = blockIdx.x * blockDim.x,
		y_b = blockIdx.y * blockDim.y,
		x = x_b + threadIdx.x,
		y = y_b + threadIdx.y,
		threadperblock = blockDim.x * blockDim.y;
	if (x >= heightmap.FreeSlipRange.x || y >= heightmap.FreeSlipRange.y) {
		return;
	}

	const uint2& dimension = heightmap.FreeSlipRange;
	auto clamp = []__device__(int val, int lower, int upper) -> int {
		return max(lower, min(val, upper));
	};
	auto float2short = []__device__(float input) -> unsigned short {
		return static_cast<unsigned short>(input * 65535u);
	};
	 
	//Cache heightmap the current thread block needs since each pixel is accessed upto 9 times.
	extern __shared__ float heightmapCache[];
	//each thread needs to access a 3x3 matrix around the current pixel, so we need to take the edge into consideration
	const uint2 cacheSize = make_uint2(blockDim.x + 2u, blockDim.y + 2u);
	unsigned int iteration = 0u;
	const unsigned int cacheSize_total = cacheSize.x * cacheSize.y;

	//TODO: Verify after cache optimisation the result is consistent with that of before the optimisation
	while (iteration < cacheSize_total) {
		const unsigned int cacheIdx = (threadIdx.x + blockDim.x * threadIdx.y) + iteration;
		const unsigned int x_w = x_b + static_cast<unsigned int>(fmodf(cacheIdx, cacheSize.x)),
			y_w = y_b + static_cast<unsigned int>(floorf(1.0f * cacheIdx / cacheSize.x));
		const unsigned int workerIdx = clamp((x_w - 1u), 0, dimension.x - 1u) + clamp((y_w - 1u), 0, dimension.y - 1u) * dimension.x;

		if (cacheIdx < cacheSize_total) {
			//make sure index don't get out of bound
			//start caching from (x-1, y-1) until (x+1, y+1)
			heightmapCache[cacheIdx] = heightmap[workerIdx];
		}
		//warp around to reuse some threads to finish all compute
		iteration += threadperblock;
	}
	__syncthreads();

	if ((heightmap.FreeSlipChunk.x * heightmap.FreeSlipChunk.y) > 1 && (x == 0 || y == 0 || x == heightmap.FreeSlipRange.x - 1 || y == heightmap.FreeSlipRange.y - 1)) {
		//if freeslip is not turned on, we need to calculate the edge pixel for this chunk
		//otherwise, do not touch the border pixel since border pixel is calculated seamlessly by other chunks
		return;
	}
	//load the cells from heightmap, remember the height map only contains one color channel
	//using Sobel fitering
	//Cache index
	const unsigned int x_c = threadIdx.x + 1u,
		y_c = threadIdx.y + 1u;
	float cell[8];
	cell[0] = heightmapCache[(x_c - 1) + (y_c - 1) * cacheSize.x];
	cell[1] = heightmapCache[x_c + (y_c - 1) * cacheSize.x];
	cell[2] = heightmapCache[(x_c + 1) + (y_c - 1) * cacheSize.x];
	cell[3] = heightmapCache[(x_c - 1) + y_c * cacheSize.x];
	cell[4] = heightmapCache[(x_c + 1) + y_c * cacheSize.x];
	cell[5] = heightmapCache[(x_c - 1) + (y_c + 1) * cacheSize.x];
	cell[6] = heightmapCache[x_c + (y_c + 1) * cacheSize.x];
	cell[7] = heightmapCache[(x_c + 1) + (y_c + 1) * cacheSize.x];
	//apply the filtering kernel matrix
	float3 normal;
	normal.z = 1.0f / strength;
	normal.x = cell[0] + 2 * cell[3] + cell[5] - (cell[2] + 2 * cell[4] + cell[7]);
	normal.y = cell[0] + 2 * cell[1] + cell[2] - (cell[5] + 2 * cell[6] + cell[7]);
	//normalize
	normal = normalize3DKERNEL(normal);
	//clamp to [0,1], was [-1,1]
	normal.x = __saturatef((normal.x + 1.0f) / 2.0f);
	normal.y = __saturatef((normal.y + 1.0f) / 2.0f);
	normal.z = __saturatef((normal.z + 1.0f) / 2.0f);
	
	//copy to the output, RGBA32F
	const unsigned int index = heightmap(x + y * dimension.x);
	heightfield[index * 4] = float2short(normal.x);//R
	heightfield[index * 4 + 1] = float2short(normal.y);//G
	heightfield[index * 4 + 2] = float2short(normal.z);//B
	heightfield[index * 4 + 3] = float2short(heightmapCache[x_c + y_c * cacheSize.x]);//A
}

__global__ void initGlobalLocalIndexKERNEL(unsigned int* output, unsigned int rowCount, uint2 chunkRange, uint2 tableSize, uint2 mapSize) {
	//current pixel
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y,
		globalidx = x + y * rowCount;
	if (x >= tableSize.x || y >= tableSize.y) {
		return;
	}

	//simple maths
	const uint2 globalPos = make_uint2(globalidx - floorf(globalidx / rowCount) * rowCount, floorf(globalidx / rowCount));
	const uint2 chunkPos = make_uint2(floorf(globalPos.x / mapSize.x), floorf(globalPos.y / mapSize.y));
	const uint2 localPos = make_uint2(globalPos.x - chunkPos.x * mapSize.x, globalPos.y - chunkPos.y * mapSize.y);

	output[globalidx] = (chunkPos.x + chunkRange.x * chunkPos.y) * mapSize.x * mapSize.y + (localPos.x + mapSize.x * localPos.y);
}