#pragma once
#include "STPHeightfieldGenerator.cuh"

#include <memory>

//TODO: remove constant memory as it's not very useful and static initialisation sucks
__constant__ unsigned char HeightfieldSettings[sizeof(SuperTerrainPlus::STPSettings::STPHeightfieldSettings)];
static unsigned int ErosionBrushSize;

using namespace SuperTerrainPlus::STPCompute;

using std::list;
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
*/
__global__ void curandInitKERNEL(STPHeightfieldGenerator::curandRNG*, unsigned long long);

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
*/
__global__ void performErosionKERNEL(STPRainDrop::STPFreeSlipManager, STPHeightfieldGenerator::curandRNG*);

/**
 * @brief Fix the edge of the central chunk such that it's aligned with all neighbour chunks
 * @param heightmap The heightmap with global-local converter
 * @param interpolation_table The index base and range defined as the central chunk
 * @param threadSum The number thread required for interpolation, it equals to the size of the interpolation table. The actual number of thread 
 * launched is usually greater than this number
*/
__global__ void performPostErosionInterpolationKERNEL(STPRainDrop::STPFreeSlipManager, const uint2*, unsigned int);

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
 * @param mapSize The dimension of the map
*/
__global__ void initGlobalLocalIndexKERNEL(unsigned int*, unsigned int, uint2, uint2);

/**
 * @brief Generate a new interpolation index table
 * @param output The generated table which contains index
 * @param chunkRange The number of chunk, it usually is the free slip range
 * @param mapSize The dimension of each map in the chunk
 * @param threadEdge Number of thread that covers edge interpolation
 * @param threadSum Number of thread in total that should be used for interpolation
*/
__global__ void initInterpolationIndexKERNEL(uint2*, uint2, uint2, unsigned int, unsigned int);

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
__host__ bool blockcopy_d2h(list<T*>& dest, T* host, T* device, size_t block_size, size_t individual_size, size_t element_count, cudaStream_t stream) {
	bool no_error = true;
	no_error &= cudaSuccess == cudaMemcpyAsync(host, device, block_size, cudaMemcpyDeviceToHost, stream);
	unsigned int base = 0u;
	for (T* map : dest) {
		no_error &= cudaSuccess == cudaMemcpyAsync(map, host + base, individual_size, cudaMemcpyHostToHost, stream);
		base += element_count;
	}

	return no_error;
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
__host__ bool blockcopy_h2d(T* device, T* host, list<T*>& source, size_t block_size, size_t individual_size, size_t element_count, cudaStream_t stream) {
	bool no_error = true;
	unsigned int base = 0u;
	for (T* map : source) {
		no_error &= cudaSuccess == cudaMemcpyAsync(host + base, map, individual_size, cudaMemcpyHostToHost, stream);
		base += element_count;
	}
	no_error &= cudaSuccess == cudaMemcpyAsync(device, host, block_size, cudaMemcpyHostToDevice, stream);

	return no_error;
}

__host__ void* STPHeightfieldGenerator::STPHeightfieldAllocator::allocate(size_t count) {
	void* mem = nullptr;
	cudaMalloc(&mem, count);
	return mem;
}

__host__ void STPHeightfieldGenerator::STPHeightfieldAllocator::deallocate(size_t count, void* ptr) {
	cudaFree(ptr);
}

__host__ void* STPHeightfieldGenerator::STPHeightfieldHostAllocator::allocate(size_t count) {
	void* mem = nullptr;
	cudaMallocHost(&mem, count);
	return mem;
}

__host__ void STPHeightfieldGenerator::STPHeightfieldHostAllocator::deallocate(size_t count, void* ptr) {
	cudaFreeHost(ptr);
}

__host__ void* STPHeightfieldGenerator::STPHeightfieldNonblockingStreamAllocator::allocate(size_t count) {
	cudaStream_t stream;
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	return reinterpret_cast<void*>(stream);
}

__host__ void STPHeightfieldGenerator::STPHeightfieldNonblockingStreamAllocator::deallocate(size_t count, void* stream) {
	cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
}

__host__ STPHeightfieldGenerator::STPHeightfieldGenerator(const STPSettings::STPSimplexNoiseSettings* noise_settings, const STPSettings::STPChunkSettings* chunk_settings) 
	: simplex_h(noise_settings), Noise_Settings(*noise_settings), FreeSlipChunk(make_uint2(chunk_settings->FreeSlipChunk.x, chunk_settings->FreeSlipChunk.y)) {
	//allocating space
	cudaMalloc(&this->simplex, sizeof(STPSimplexNoise));
	//copy data
	cudaMemcpy(this->simplex, &simplex_h, sizeof(STPSimplexNoise), cudaMemcpyHostToDevice);

	//TODO: use CUDA API to dynamically determine launch configuration
	//kernel parameters
	this->numThreadperBlock_Map = dim3(32u, 32u);
	this->numBlock_Map = dim3(noise_settings->Dimension.x / numThreadperBlock_Map.x, noise_settings->Dimension.y / numThreadperBlock_Map.y);
	this->numThreadperBlock_Erosion = 1024u;
	this->numThreadperBlock_Interpolation = 1024u;

	//set global local index
	this->initLocalGlobalIndexCUDA();
	//set interpolation index
	this->initInterpolationIndexCUDA();
}

__host__ STPHeightfieldGenerator::~STPHeightfieldGenerator() {
	//TODO: maybe use a smart ptr with custom deleter?
	cudaFree(this->simplex);
	//check if the rng has been init
	if (this->RNG_Map != nullptr) {
		cudaFree(this->RNG_Map);
	}
	if (this->BiomeDictionary != nullptr) {
		cudaFree(this->BiomeDictionary);
	}
	if (this->GlobalLocalIndex != nullptr) {
		cudaFree(this->GlobalLocalIndex);
	}
	if (this->InterpolationIndex != nullptr) {
		cudaFree(this->InterpolationIndex);
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
	return cudaSuccess == cudaMemcpyToSymbol(HeightfieldSettings, stored_settings.get(), sizeof(STPSettings::STPHeightfieldSettings), 0ull, cudaMemcpyHostToDevice);
}

__host__ bool STPHeightfieldGenerator::operator()(STPMapStorage& args, STPGeneratorOperation operation) const {
	//check the availiability of the engine
	if (this->RNG_Map == nullptr) {
		return false;
	}
	//check the availability of biome dictionary
	/*if (this->BiomeDictionary == nullptr) {
		return false;
	}*/
	if (operation == 0u) {
		//no operation is specified, nothing can be done
		return false;
	}

	bool no_error = true;//check for error, true if all successful
	//TODO: there is no need to recompute constants repeatedly, store them in the class object
	//allocating spaces for texture, storing on device
	//this is the size for a texture in one channel
	const unsigned int num_pixel = this->Noise_Settings.Dimension.x * this->Noise_Settings.Dimension.y;
	const unsigned int map_size = num_pixel * sizeof(float);
	const unsigned int map16ui_size = num_pixel * sizeof(unsigned short) * 4u;
	//if free-slip erosion is disabled, it should be one.
	//if enabled, it should be the product of two dimension in free-slip chunk.
	const unsigned int freeslip_chunk_total = args.Heightmap32F.size();
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

	//memory allocation
	//Device
	{
		unique_lock<mutex> lock(this->MapCacheDevice_lock);
		//FP32
		//we need heightmap for computation regardlessly
		heightfield_freeslip_d = reinterpret_cast<float*>(this->MapCacheDevice.allocate(map_freeslip_size));
		//INT16
		if (flag[2]) {
			heightfield_formatted_d = reinterpret_cast<unsigned short*>(this->MapCacheDevice.allocate(map16ui_freeslip_size));
		}
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

	//setup phase
	//creating stream so cpu thread can calculate all chunks altogether
	cudaStream_t stream = nullptr;
	//we want the stream to not be blocked by default stream
	{
		unique_lock<mutex> stream_lock(this->StreamPool_lock);
		stream = reinterpret_cast<cudaStream_t>(this->StreamPool.allocate(1ull));
	}
	
	//Flag: HeightmapGeneration
	if (flag[0]) {
		//generate a new heightmap and store it to the output later
		generateHeightmapKERNEL << <this->numBlock_Map, this->numThreadperBlock_Map, 0, stream >> > (this->simplex, heightfield_freeslip_d,
			this->Noise_Settings.Dimension, make_float2(1.0f * this->Noise_Settings.Dimension.x / 2.0f, 1.0f * this->Noise_Settings.Dimension.y / 2.0f), args.HeightmapOffset);
	}
	else {
		//no generation, use existing
		no_error &= blockcopy_h2d(heightfield_freeslip_d, heightfield_freeslip_h, args.Heightmap32F, map_freeslip_size, map_size, num_pixel, stream);
	}

	if (flag[1] || flag[2]) {
		const STPRainDrop::STPFreeSlipManager heightmap_slip(heightfield_freeslip_d, this->GlobalLocalIndex, this->FreeSlipChunk, this->Noise_Settings.Dimension);

		//Flag: Erosion
		if (flag[1]) {
			//erode the heightmap, either from provided heightmap or generated previously
			const unsigned erosionBrushCache_size = ErosionBrushSize * (sizeof(int) + sizeof(float));
			performErosionKERNEL << <this->numBlock_Erosion, this->numThreadperBlock_Erosion, erosionBrushCache_size, stream >> > (heightmap_slip, this->RNG_Map);
			//check if free slip has been enabled
			if (this->InterpolationThreadRequired > 0u) {
				//perform post erosion interpolation using interpolation lookup table
				performPostErosionInterpolationKERNEL << <this->numBlock_Interpolation, this->numThreadperBlock_Interpolation, 0, stream >> > (heightmap_slip, this->InterpolationIndex, this->InterpolationThreadRequired);
			}
		}

		//Flag: RenderingBufferGeneration
		if (flag[2]) {
			//generate normalmap from heightmap and format into rendering buffer
			const unsigned int cacheSize = (this->numThreadperBlock_Map.x + 2u) * (this->numThreadperBlock_Map.y + 2u) * sizeof(float);
			generateRenderingBufferKERNEL << <this->numBlock_FreeslipMap, this->numThreadperBlock_Map, cacheSize, stream >> > (heightmap_slip, heightfield_formatted_d);
		}
	}
	
	
	//Store the result accordingly
	//copy the result back to the host
	//heightmap will always be available
	if (flag[0] || flag[1]) {
		//copy all heightmap chunks back if heightmap has been modified
		no_error &= blockcopy_d2h(args.Heightmap32F, heightfield_freeslip_h, heightfield_freeslip_d, map_freeslip_size, map_size, num_pixel, stream);
	}
	//copy the rendering buffer result if enabled
	if (flag[2]) {
		//copy heightfield
		no_error &= blockcopy_d2h(args.Heightfield16UI, heightfield_formatted_h, heightfield_formatted_d, map16ui_freeslip_size, map16ui_size, num_pixel * 4u, stream);
	}

	//waiting for finish
	no_error &= cudaSuccess == cudaStreamSynchronize(stream);
	{
		unique_lock<mutex> stream_lock(this->StreamPool_lock);
		this->StreamPool.deallocate(1ull, reinterpret_cast<void*>(stream));
	}

	//Finish up the rest, clear up when the device is ready
	//nullptr means not allocated
	{
		unique_lock<mutex> lock(this->MapCacheDevice_lock);
		if (heightfield_freeslip_d != nullptr) {
			this->MapCacheDevice.deallocate(map_freeslip_size, heightfield_freeslip_d);
		}
		if (heightfield_formatted_d != nullptr) {
			this->MapCacheDevice.deallocate(map16ui_freeslip_size, heightfield_formatted_d);
		}
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

	return no_error;
}

__host__ bool STPHeightfieldGenerator::setErosionIterationCUDA(unsigned int raindrop_count) {
	//set the launch parameter
	this->numBlock_Erosion = raindrop_count / this->numThreadperBlock_Erosion;
	bool no_error = true;

	//make sure all previous takes are finished
	no_error &= cudaSuccess == cudaDeviceSynchronize();
	//when the raindrop count changes, we need to reallocate and regenerate the rng
	//the the number of rng = the number of the raindrop
	//such that each raindrop has independent rng
	//allocating spaces for rng storage array
	if (this->RNG_Map != nullptr) {
		//if there is an old version existing, we need to delete the old one
		no_error &= cudaSuccess == cudaFree(this->RNG_Map);
	}
	no_error &= cudaSuccess == cudaMalloc(&this->RNG_Map, sizeof(curandRNG) * raindrop_count);
	//and send to kernel
	curandInitKERNEL << <this->numBlock_Erosion, this->numThreadperBlock_Erosion >> > (this->RNG_Map, this->Noise_Settings.Seed);
	no_error &= cudaSuccess == cudaDeviceSynchronize();
	//leave the result on device, and update the raindrop count
	this->NumRaindrop = raindrop_count;

	return no_error;
}

__host__ unsigned int STPHeightfieldGenerator::getErosionIteration() const {
	return this->NumRaindrop;
}

__host__ void STPHeightfieldGenerator::initLocalGlobalIndexCUDA() {
	const uint2& dimension = this->Noise_Settings.Dimension;
	const uint2& range = this->FreeSlipChunk;
	const uint2 global_dimension = make_uint2(range.x * dimension.x, range.y * dimension.y);
	//launch parameters
	this->numBlock_FreeslipMap = dim3(global_dimension.x / this->numThreadperBlock_Map.x, global_dimension.y / this->numThreadperBlock_Map.y);
	//Don't generate the table when FreeSlipChunk.xy are both 1, and in STPRainDrop don't use the table
	if (range.x == 1u && range.y == 1u) {
		this->GlobalLocalIndex = nullptr;
		return;
	}

	bool no_error = true;
	//make sure all previous takes are finished
	no_error &= cudaSuccess == cudaDeviceSynchronize();
	//allocation
	no_error &= cudaSuccess == cudaMalloc(&this->GlobalLocalIndex, sizeof(unsigned int) * global_dimension.x * global_dimension.y);
	//compute
	initGlobalLocalIndexKERNEL << <this->numBlock_FreeslipMap, this->numThreadperBlock_Map >> > (this->GlobalLocalIndex, global_dimension.x, range, dimension);
	no_error &= cudaSuccess == cudaDeviceSynchronize();

	if (!no_error) {
		throw std::runtime_error("Heightfield generator initialisation failed::Could not initialise local global index table");
	}
}

__host__ void STPHeightfieldGenerator::initInterpolationIndexCUDA() {
	const uint2& dimension = this->Noise_Settings.Dimension;
	const uint2& range = this->FreeSlipChunk;
	//Disable interpolation when freeslip erosion is not enabled
	if (range.x == 1u && range.y == 1u) {
		this->InterpolationThreadRequired = 0u;
		return;
	}

	//Number of thread that covers edge interpolation
	const unsigned int threadEdge = (dimension.x - 2u) * (range.x * (range.x - 1u)) + (dimension.y - 2u) * (range.y * (range.y - 1u));
	//Number of thread that covers corner interpolation
	const unsigned int threadCorner = (range.x - 1u) * (range.y - 1u);
	//Total number of thread needed for all interpolation, note that the actual number of thread launched might be more than this number
	this->InterpolationThreadRequired = threadEdge + threadCorner;
	//launch parameters
	this->numBlock_Interpolation = static_cast<unsigned int>(ceilf(1.0f * this->InterpolationThreadRequired / this->numThreadperBlock_Interpolation));
	bool no_error = true;

	no_error &= cudaSuccess == cudaDeviceSynchronize();
	//allocation
	no_error &= cudaSuccess == cudaMalloc(&this->InterpolationIndex, sizeof(uint2) * this->InterpolationThreadRequired);
	//compute
	initInterpolationIndexKERNEL << <this->numBlock_Interpolation, this->numThreadperBlock_Interpolation >> > (this->InterpolationIndex, range, dimension, threadEdge, this->InterpolationThreadRequired);
	no_error &= cudaSuccess == cudaDeviceSynchronize();

	if (!no_error) {
		throw std::runtime_error("Heightfield generator initialisation failed::Could not initialise interpolation index table");
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

__global__ void curandInitKERNEL(STPHeightfieldGenerator::curandRNG* rng, unsigned long long seed) {
	//current working index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	//the same seed but we are looking for different sequence
	curand_init(seed, static_cast<unsigned long long>(index), 0, &rng[index]);
}

__global__ void generateHeightmapKERNEL(STPSimplexNoise* const noise_fun, float* height_storage,
	uint2 dimension, float2 half_dimension, float3 offset) {
	//convert constant memory to usable class
	const SuperTerrainPlus::STPSettings::STPHeightfieldSettings* const settings = reinterpret_cast<const SuperTerrainPlus::STPSettings::STPHeightfieldSettings* const>(HeightfieldSettings);

	//the current working pixel
	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y;
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

__global__ void performErosionKERNEL(STPRainDrop::STPFreeSlipManager heightmap_storage, STPHeightfieldGenerator::curandRNG* rng) {
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

	//current working index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
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

__global__ void performPostErosionInterpolationKERNEL(STPRainDrop::STPFreeSlipManager heightmap, const uint2* interpolation_table, unsigned int threadSum) {
	//current working thread, 1D
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	//retrieve the MSB value and set MSB of the original value back to zero
	auto decodeMSB = []__device__(unsigned int& value) -> bool {
		constexpr unsigned int shifted = static_cast<unsigned int>(sizeof(value) * CHAR_BIT - 1u);
		const bool msb = ((value >> shifted) & 0b1u) == 1u;
		value = value & ~(1u << shifted);
		return msb;
	};

	//kill the worker if it's more than enough
	if (tid >= threadSum) {
		return;
	}

	//read the coordinate to interpolate
	uint2 location = interpolation_table[tid];
	//decode
	const bool flag[2] = { decodeMSB(location.x), decodeMSB(location.y) };

	float interpolated;
	const unsigned int rowCount = heightmap.FreeSlipRange.x;
	const unsigned int index = location.x + rowCount * location.y;
	if (flag[0] && flag[1]) {
		//corner interpolation
		//find mean in a 2x2 block, start from the top-left pixel
		interpolated = heightmap[index] + heightmap[index + 1u] + heightmap[index + rowCount] + heightmap[index + rowCount + 1u];
		interpolated /= 4.0f;
		//assigning interpolated value
		heightmap[index] = interpolated;
		heightmap[index + 1u] = interpolated;
		heightmap[index + rowCount] = interpolated;
		heightmap[index + rowCount + 1u] = interpolated;
		return;
	}

	if (flag[0] && !flag[1]) {
		//column edge interpolation
		//find mean for pixels from left to right
		interpolated = heightmap[index] + heightmap[index + 1u];
		interpolated /= 2.0f;
		//assigning values
		heightmap[index] = interpolated;
		heightmap[index + 1u] = interpolated;
		return;
	}
	if (!flag[0] && flag[1]) {
		//row edge interpolation
		//find mean for pixels from up to down
		interpolated = heightmap[index] + heightmap[index + rowCount];
		interpolated /= 2.0f;
		//assigning values
		heightmap[index] = interpolated;
		heightmap[index + rowCount] = interpolated;
		return;
	}

	//otherwise it's an error
	assert("Invalid flag identifier");
}

__global__ void generateRenderingBufferKERNEL(STPRainDrop::STPFreeSlipManager heightmap, unsigned short* heightfield) {
	//convert constant memory to usable class
	SuperTerrainPlus::STPSettings::STPHeightfieldSettings* const settings = reinterpret_cast<SuperTerrainPlus::STPSettings::STPHeightfieldSettings* const>(HeightfieldSettings);
	//the current working pixel
	const unsigned int x_b = blockIdx.x * blockDim.x,
		y_b = blockIdx.y * blockDim.y,
		x = x_b + threadIdx.x,
		y = y_b + threadIdx.y,
		threadperblock = blockDim.x * blockDim.y;
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
		const unsigned int workerIdx = clamp((x_w - 1u), 0, dimension.x - 1u) + clamp((y_w - 1u), 0, dimension.x - 1u) * dimension.x;

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
	
	return;
}

__global__ void initGlobalLocalIndexKERNEL(unsigned int* output, unsigned int rowCount, uint2 chunkRange, uint2 mapSize) {
	//current pixel
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y,
		globalidx = x + y * rowCount;
	//simple maths
	const uint2 globalPos = make_uint2(globalidx - floorf(globalidx / rowCount) * rowCount, floorf(globalidx / rowCount));
	const uint2 chunkPos = make_uint2(floorf(globalPos.x / mapSize.x), floorf(globalPos.y / mapSize.y));
	const uint2 localPos = make_uint2(globalPos.x - chunkPos.x * mapSize.x, globalPos.y - chunkPos.y * mapSize.y);

	output[globalidx] = (chunkPos.x + chunkRange.x * chunkPos.y) * mapSize.x * mapSize.y + (localPos.x + mapSize.x * localPos.y);
}

__global__ void initInterpolationIndexKERNEL(uint2* output, uint2 chunkRange, uint2 mapSize, unsigned int threadEdge, unsigned int threadSum) {
	//pre-compute constant values
	//An iteration contains (chunkRange.x - 1) column edges and (chunkRange.x) row edges.
	//In a complete chunk there will always be (chunkRange.y - 1) complete iterations plus an incomplete iteration with (chunkRange.x - 1) column edges
	//The max index can reach for column edge per iteration
	__shared__ unsigned int maxIndex_column;
	//Sum of maxIndex_column and maxIndex_row
	__shared__ unsigned int maxIndex;
	if (threadIdx.x == 0u) {
		maxIndex_column = (mapSize.y - 2u) * (chunkRange.x - 1u);
		//The max index can reach for row edge per iteration
		const unsigned int maxIndex_row = (mapSize.x - 2u) * (chunkRange.x);
		maxIndex = maxIndex_column + maxIndex_row;
	}
	__syncthreads();

	//current thread id, it's a 1D launch config
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	//We will encode MSB to tell what type of interpolation it's
	auto setMSB = []__device__(unsigned int value, bool bit) -> unsigned int {
		constexpr unsigned int flag = 1u << static_cast<unsigned int>(sizeof(value) * CHAR_BIT - 1u);
		return bit ? value | flag : value & ~flag;
	};
	
	//checking for index out of bound
	if (tid >= threadSum) {
		return;
	}

	//the power of maths, again
	if (tid >= threadEdge) {
		//corner interpolation
		const unsigned int corner_index = tid - threadEdge;
		output[tid].x = mapSize.x * (corner_index % (chunkRange.x - 1u) + 1u) - 1u;
		output[tid].y = mapSize.y * (static_cast<unsigned int>(floorf(1.0f * corner_index / (chunkRange.x - 1u))) + 1u) - 1u;

		//binary insertion for corner: MSB.x = MSB.y = 1
		output[tid].x = setMSB(output[tid].x, true);
		output[tid].y = setMSB(output[tid].y, true);
		return;
	}

	//edge interpolation
	if (tid % maxIndex < maxIndex_column) {
		//column edge interpolation
		const unsigned int column_index = tid % (chunkRange.x - 1u);
		output[tid].x = mapSize.x - 1u + (column_index * mapSize.x);
		output[tid].y = (static_cast<unsigned int>(floorf(1.0f * (tid % maxIndex) / (chunkRange.x - 1u))) + 1u)
			+ mapSize.y * static_cast<unsigned int>(floorf(1.0f * tid / maxIndex));

		//binary insertion for column edge: MSB.x = 1 and MSB.y = 0
		output[tid].x = setMSB(output[tid].x, true);
		output[tid].y = setMSB(output[tid].y, false);
		return;
	}
	//row edge interpolation
	const unsigned int current_row_index = tid % maxIndex - maxIndex_column;
	const unsigned int row_index = static_cast<unsigned int>(floorf(1.0f * current_row_index / (mapSize.x - 2u)));
	output[tid].x = current_row_index % (mapSize.x - 2u) + row_index * mapSize.x + 1u;
	output[tid].y = mapSize.y * (static_cast<unsigned int>(floorf(1.0f * tid / maxIndex)) + 1u) - 1u;
	
	//binary insertion for row edge: MSB.x = 0 and MSB.y = 1
	output[tid].x = setMSB(output[tid].x, false);
	output[tid].y = setMSB(output[tid].y, true);
}