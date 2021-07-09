#pragma once
#include "STPHeightfieldGenerator.cuh"

#define STP_EXCEPTION_ON_ERROR
#include "STPDeviceErrorHandler.cuh"

#include <memory>

//TODO: remove constant memory as it's not very useful and static initialisation sucks
__constant__ unsigned char HeightfieldSettings[sizeof(SuperTerrainPlus::STPSettings::STPHeightfieldSettings)];
static unsigned int ErosionBrushSize;

using namespace SuperTerrainPlus::STPCompute;

using std::vector;
using std::mutex;
using std::unique_lock;
using std::unique_ptr;
using std::make_unique;

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
 * @param height_storage - The pointer to a location where the heightmap will be stored
 * @param dimension - The width and height of the generated heightmap
 * @param half_dimension - Precomputed dimension/2 so the kernel don't need to repeatly compute that
 * @param offset - Controlling the offset on x, y and height offset on z
*/
__global__ void generateHeightmapKERNEL(STPSimplexNoise* const, float*, uint2, float2, float3);

/**
 * @brief Performing hydraulic erosion for the given heightmap terrain using CUDA parallel computing
 * @param height_storage The heightmap with global-local convertion management
 * @param rng The random number generator map sequence, independent for each rain drop
 * @param raindrop_count The expected number of raindrop
*/
__global__ void performErosionKERNEL(STPRainDrop::STPFreeSlipManager, STPHeightfieldGenerator::curandRNG*, unsigned int);

/**
 * @brief Generate the normal map for the height map within kernel, and combine two maps into a rendering buffer
 * @param heightmap - contains the height map that will be used to generate the normalmap, with free-slip manager
 * @param heightfield - will be used to store the output of the normal map in RGB channel, heightmap will be copied to A channel
*/
__global__ void generateRenderingBufferKERNEL(STPRainDrop::STPFreeSlipManager, unsigned short*);

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
__host__ void blockcopy_d2h(const vector<T*>& dest, T* host, T* device, size_t block_size, size_t individual_size, size_t element_count, cudaStream_t stream) {
	STPcudaCheckErr(cudaMemcpyAsync(host, device, block_size, cudaMemcpyDeviceToHost, stream));
	unsigned int base = 0u;
	for (T* map : dest) {
		STPcudaCheckErr(cudaMemcpyAsync(map, host + base, individual_size, cudaMemcpyHostToHost, stream));
		base += element_count;
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
	for (T* map : source) {
		STPcudaCheckErr(cudaMemcpyAsync(host + base, map, individual_size, cudaMemcpyHostToHost, stream));
		base += element_count;
	}
	STPcudaCheckErr(cudaMemcpyAsync(device, host, block_size, cudaMemcpyHostToDevice, stream));

}

__host__ void* STPHeightfieldGenerator::STPHeightfieldHostAllocator::allocate(size_t count) {
	void* mem = nullptr;
	STPcudaCheckErr(cudaMallocHost(&mem, count));
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

__host__ STPHeightfieldGenerator::STPHeightfieldGenerator(const STPSettings::STPSimplexNoiseSettings* noise_settings, const STPSettings::STPChunkSettings* chunk_settings, unsigned int hint_level_of_concurrency)
	: simplex_h(noise_settings), Noise_Settings(*noise_settings), FreeSlipChunk(make_uint2(chunk_settings->FreeSlipChunk.x, chunk_settings->FreeSlipChunk.y)) {
	const unsigned int num_pixel = chunk_settings->MapSize.x * chunk_settings->MapSize.y,
		num_freeslip_chunk = chunk_settings->FreeSlipChunk.x * chunk_settings->FreeSlipChunk.y;

	//allocating space
	STPcudaCheckErr(cudaMalloc(&this->simplex, sizeof(STPSimplexNoise)));
	//copy data
	STPcudaCheckErr(cudaMemcpy(this->simplex, &simplex_h, sizeof(STPSimplexNoise), cudaMemcpyHostToDevice));
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

	//set global local index
	this->initLocalGlobalIndexCUDA();
}

__host__ STPHeightfieldGenerator::~STPHeightfieldGenerator() {
	//TODO: maybe use a smart ptr with custom deleter?
	STPcudaCheckErr(cudaFree(this->simplex));
	STPcudaCheckErr(cudaMemPoolDestroy(this->MapCacheDevice));
	//check if the rng has been init
	if (this->RNG_Map != nullptr) {
		STPcudaCheckErr(cudaFree(this->RNG_Map));
	}
	if (this->BiomeDictionary != nullptr) {
		STPcudaCheckErr(cudaFree(this->BiomeDictionary));
	}
	if (this->GlobalLocalIndex != nullptr) {
		STPcudaCheckErr(cudaFree(this->GlobalLocalIndex));
	}
}

__host__ bool STPHeightfieldGenerator::InitGenerator(const STPSettings::STPHeightfieldSettings* const settings) {
	//keep a local copy of the setting so device can have access to the pointer inside the class
	static unique_ptr<const STPSettings::STPHeightfieldSettings> stored_settings;

	//if memory address isn't the same
	if (stored_settings.get() != settings) {
		//validate memory
		if (!settings->validate()) {
			return false;
		}
		//replace current settings
		//deep copy the thing
		stored_settings = make_unique<const STPSettings::STPHeightfieldSettings>(*settings);
	}

	ErosionBrushSize = stored_settings->getErosionBrushSize();
	STPcudaCheckErr(cudaMemcpyToSymbol(HeightfieldSettings, stored_settings.get(), sizeof(STPSettings::STPHeightfieldSettings), 0ull, cudaMemcpyHostToDevice));
	return true;
}

__host__ void STPHeightfieldGenerator::operator()(STPMapStorage& args, STPGeneratorOperation operation) const {
	//check the availiability of the engine
	if (this->RNG_Map == nullptr) {
		return;
	}
	//check the availability of biome dictionary
	/*if (this->BiomeDictionary == nullptr) {
		return false;
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
	const bool flag[3] = {
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
	if (flag[2]) {
		STPcudaCheckErr(cudaMallocFromPoolAsync(&heightfield_formatted_d, map16ui_freeslip_size, this->MapCacheDevice, stream));
	}
	//Host
	{
		unique_lock<mutex> lock(this->MapCachePinned_lock);
		//FP32
		heightfield_freeslip_h = reinterpret_cast<float*>(this->MapCachePinned.allocate(map_freeslip_size));
		//INT16
		if (flag[2]) {
			heightfield_formatted_h = reinterpret_cast<unsigned short*>(this->MapCachePinned.allocate(map16ui_freeslip_size));
		}
	}
	
	//Flag: HeightmapGeneration
	if (flag[0]) {
		STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &generateHeightmapKERNEL));
		Dimblocksize = dim3(32, blocksize / 32);
		Dimgridsize = dim3((this->Noise_Settings.Dimension.x + Dimblocksize.x - 1) / Dimblocksize.x, (this->Noise_Settings.Dimension.y + Dimblocksize.y - 1) / Dimblocksize.y);

		//generate a new heightmap and store it to the output later
		generateHeightmapKERNEL << <Dimgridsize, Dimblocksize, 0, stream >> > (this->simplex, heightfield_freeslip_d,
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

	if (flag[1] || flag[2]) {
		const STPRainDrop::STPFreeSlipManager heightmap_slip(heightfield_freeslip_d, this->GlobalLocalIndex, this->FreeSlipChunk, this->Noise_Settings.Dimension);

		//Flag: Erosion
		if (flag[1]) {
			const unsigned erosionBrushCache_size = ErosionBrushSize * (sizeof(int) + sizeof(float));
			STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &performErosionKERNEL, erosionBrushCache_size));
			gridsize = (this->NumRaindrop + blocksize - 1) / blocksize;

			//erode the heightmap, either from provided heightmap or generated previously
			performErosionKERNEL << <gridsize, blocksize, erosionBrushCache_size, stream >> > (heightmap_slip, this->RNG_Map, this->NumRaindrop);
			STPcudaCheckErr(cudaGetLastError());
		}

		//Flag: RenderingBufferGeneration
		if (flag[2]) {
			auto det_cacheSize = []__host__ __device__(int blockSize) -> size_t {
				return (blockSize + 2u) * sizeof(float);
			};
			STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&Mingridsize, &blocksize, &generateRenderingBufferKERNEL, det_cacheSize));
			Dimblocksize = dim3(32, blocksize / 32);
			Dimgridsize = dim3((freeslip_dimension.x + Dimblocksize.x - 1) / Dimblocksize.x, (freeslip_dimension.y + Dimblocksize.y - 1) / Dimblocksize.y);

			//generate normalmap from heightmap and format into rendering buffer
			const unsigned int cacheSize = (Dimblocksize.x + 2u) * (Dimblocksize.y + 2u) * sizeof(float);
			generateRenderingBufferKERNEL << <Dimgridsize, Dimblocksize, cacheSize, stream >> > (heightmap_slip, heightfield_formatted_d);
			STPcudaCheckErr(cudaGetLastError());
		}
	}
	
	//Store the result accordingly
	//copy the result back to the host
	//heightmap will always be available
	try {
		if (flag[0] || flag[1]) {
			//copy all heightmap chunks back if heightmap has been modified
			blockcopy_d2h(args.Heightmap32F, heightfield_freeslip_h, heightfield_freeslip_d, map_freeslip_size, map_size, num_pixel, stream);
		}
		//copy the rendering buffer result if enabled
		if (flag[2]) {
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

__host__ void STPHeightfieldGenerator::setErosionIterationCUDA(unsigned int raindrop_count) {
	//determine launch parameters
	int Mingridsize, gridsize, blocksize;
	STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &curandInitKERNEL));
	gridsize = (raindrop_count + blocksize - 1) / blocksize;

	//make sure all previous takes are finished
	STPcudaCheckErr(cudaDeviceSynchronize());
	//when the raindrop count changes, we need to reallocate and regenerate the rng
	//the the number of rng = the number of the raindrop
	//such that each raindrop has independent rng
	//allocating spaces for rng storage array
	if (this->RNG_Map != nullptr) {
		//if there is an old version existing, we need to delete the old one
		STPcudaCheckErr(cudaFree(this->RNG_Map));
	}
	STPcudaCheckErr(cudaMalloc(&this->RNG_Map, sizeof(curandRNG) * raindrop_count));
	//and send to kernel
	curandInitKERNEL << <gridsize, blocksize >> > (this->RNG_Map, this->Noise_Settings.Seed, raindrop_count);
	STPcudaCheckErr(cudaGetLastError());
	STPcudaCheckErr(cudaDeviceSynchronize());
	//leave the result on device, and update the raindrop count
	this->NumRaindrop = raindrop_count;
}

__host__ unsigned int STPHeightfieldGenerator::getErosionIteration() const {
	return this->NumRaindrop;
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
	STPcudaCheckErr(cudaMalloc(&this->GlobalLocalIndex, sizeof(unsigned int) * global_dimension.x * global_dimension.y));
	//compute
	initGlobalLocalIndexKERNEL << <Dimgridsize, Dimblocksize >> > (this->GlobalLocalIndex, global_dimension.x, range, global_dimension, dimension);
	STPcudaCheckErr(cudaGetLastError());
	STPcudaCheckErr(cudaDeviceSynchronize());
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

__global__ void generateHeightmapKERNEL(STPSimplexNoise* const noise_fun, float* height_storage,
	uint2 dimension, float2 half_dimension, float3 offset) {
	//the current working pixel
	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= dimension.x || y >= dimension.y) {
		return;
	}

	//convert constant memory to usable class
	const SuperTerrainPlus::STPSettings::STPHeightfieldSettings* const settings = reinterpret_cast<const SuperTerrainPlus::STPSettings::STPHeightfieldSettings* const>(HeightfieldSettings);
	float amplitude = 1.0f, frequency = 1.0f, noiseheight = 0.0f;
	float min = 0.0f, max = 0.0f;//The min and max indicates the range of the multi-phased simplex function, not the range of the output texture
	//multiple phases of noise
	for (int i = 0; i < settings->Octave; i++) {
		float sampleX = ((1.0 * x - half_dimension.x) + offset.x) / settings->Scale * frequency, //subtract the half width and height can make the scaling focus at the center
			sampleY = ((1.0 * y - half_dimension.y) + offset.z) / settings->Scale * frequency;//since the y is inverted we want to filp it over
		noiseheight += noise_fun->simplex2D(sampleX, sampleY) * amplitude;

		//calculate the min and max
		min -= 1.0f * amplitude;
		max += 1.0f * amplitude;
		//scale the parameters
		amplitude *= settings->Persistence;
		frequency *= settings->Lacunarity;
	}
	
	//interpolate and clamp the value within [0,1], was [min,max]
	noiseheight = InvlerpKERNEL(min, max, noiseheight + offset.y);
	//finally, output the texture
	height_storage[x + y * dimension.x] = noiseheight;//we have only allocated R32F format;
	
	return;
}

__global__ void performErosionKERNEL(STPRainDrop::STPFreeSlipManager heightmap_storage, STPHeightfieldGenerator::curandRNG* rng, unsigned int raindrop_count) {
	//current working index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= raindrop_count) {
		return;
	}

	//convert constant memory to usable class
	const SuperTerrainPlus::STPSettings::STPRainDropSettings* const settings = (const SuperTerrainPlus::STPSettings::STPRainDropSettings* const)(reinterpret_cast<const SuperTerrainPlus::STPSettings::STPHeightfieldSettings* const>(HeightfieldSettings));
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
	STPRainDrop droplet(initPos, settings->initWaterVolume, settings->initSpeed);
	droplet.Erode(settings, heightmap_storage);
}

__global__ void generateRenderingBufferKERNEL(STPRainDrop::STPFreeSlipManager heightmap, unsigned short* heightfield) {
	//the current working pixel
	const unsigned int x_b = blockIdx.x * blockDim.x,
		y_b = blockIdx.y * blockDim.y,
		x = x_b + threadIdx.x,
		y = y_b + threadIdx.y,
		threadperblock = blockDim.x * blockDim.y;
	if (x >= heightmap.FreeSlipRange.x || y >= heightmap.FreeSlipRange.y) {
		return;
	}

	//convert constant memory to usable class
	SuperTerrainPlus::STPSettings::STPHeightfieldSettings* const settings = reinterpret_cast<SuperTerrainPlus::STPSettings::STPHeightfieldSettings* const>(HeightfieldSettings);
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
			y_w = static_cast<unsigned int>(y_b + floorf(1.0f * cacheIdx / cacheSize.x));
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
	normal.z = 1.0f / settings->Strength;
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