#pragma once
#include "STPHeightfieldGenerator.cuh"

#include <memory>

__constant__ unsigned char HeightfieldSettings[sizeof(SuperTerrainPlus::STPSettings::STPHeightfieldSettings)];

using namespace SuperTerrainPlus::STPCompute;

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
 * @brief Clamp the input value with the range
 * @param val The clamping value
 * @param lower The lowest possible value
 * @param upper The highest possible value
 * @return val if [lower, upper], lower if val < lower, upper if val > upper
*/
__device__ __forceinline__ int clamp(int, int, int);

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
 * @brief Generate the normal map for the height map within kernel
 * @param heightmap - contains the height map that will be wused to generate the normal
 * @param normal_storage - normal map, will be used to store the output of the normal map
 * @param dimension - The width and height of both map
 * @return True if the normal map is successully generated without errors
*/
__global__ void generateNormalmapKERNEL(float* const, float*, uint2);

/**
 * @brief Convert _32F format to _16
 * @param input The input image, each color channel occupies 32 bit (float)
 * @param output The output image, each color channel occupies 16 bit (unsigne short int).
 * @param channel How many channel in the texture, the input and output channel will have the same number of channel
 * @return True if conversion was successful without errors
*/
__global__ void floatToshortKERNEL(const float* const, unsigned short*, uint2, unsigned int);

/**
 * @brief Generate a new global to local index table
 * @param output The generated table. Should be preallocated with size sizeof(unsigned int) * chunkRange.x * mapSize.x * chunkRange.y * mapSize.y
 * @param rowCount The number of row in the global index table, which is equivalent to chunkRange.x * mapSize.x
 * @param chunkRage The number of chunk (or locals)
 * @param mapSize The dimension of the map
*/
__global__ void initGlobalLocalIndexKERNEL(unsigned int*, unsigned int, uint2, uint2);

__host__ float* STPHeightfieldGenerator::STPHeightfieldAllocator::allocate(size_t count) {
	float* mem = nullptr;
	cudaMalloc(&mem, sizeof(float) * count);
	return mem;
}

__host__ void STPHeightfieldGenerator::STPHeightfieldAllocator::deallocate(size_t count, float* ptr) {
	cudaFree(ptr);
}

__host__ unsigned short* STPHeightfieldGenerator::STPImageConverterAllocator::allocate(size_t count) {
	unsigned short* mem = nullptr;
	cudaMalloc(&mem, sizeof(unsigned short) * count);
	return mem;
}

__host__ void STPHeightfieldGenerator::STPImageConverterAllocator::deallocate(size_t count, unsigned short* ptr) {
	cudaFree(ptr);
}

__host__ float* STPHeightfieldGenerator::STPHeightfieldHostAllocator::allocate(size_t count) {
	float* mem = nullptr;
	cudaMallocHost(&mem, sizeof(float) * count);
	return mem;
}

__host__ void STPHeightfieldGenerator::STPHeightfieldHostAllocator::deallocate(size_t count, float* ptr) {
	cudaFreeHost(ptr);
}

__host__ STPHeightfieldGenerator::STPHeightfieldGenerator(const STPSettings::STPSimplexNoiseSettings* noise_settings, const STPSettings::STPChunkSettings* chunk_settings) 
	: simplex_h(noise_settings), Noise_Settings(*noise_settings), FreeSlipChunk(make_uint2(chunk_settings->FreeSlipChunk.x, chunk_settings->FreeSlipChunk.y)) {
	//allocating space
	cudaMalloc(&this->simplex, sizeof(STPSimplexNoise));
	//copy data
	cudaMemcpy(this->simplex, &simplex_h, sizeof(STPSimplexNoise), cudaMemcpyHostToDevice);

	//kernel parameters
	this->numThreadperBlock_Map = dim3(32, 32);
	this->numBlock_Map = dim3(noise_settings->Dimension.x / numThreadperBlock_Map.x, noise_settings->Dimension.y / numThreadperBlock_Map.y);
	this->numThreadperBlock_Erosion = 1024;
	this->numBlock_Erosion = 0;//This will be set after user call the setErosionIterationCUDA() method

	//set global local index
	if (!this->setLocalGlobalIndexCUDA()) {
		throw std::runtime_error("Heightfield generator initialisation failed::Could not initialise local global index table");
	}
}

__host__ STPHeightfieldGenerator::~STPHeightfieldGenerator() {
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
}

__host__ bool STPHeightfieldGenerator::useSettings(const STPSettings::STPHeightfieldSettings* const settings) {
	//keep a local copy of the setting so device can have access to the pointer inside the class
	static std::unique_ptr<const STPSettings::STPHeightfieldSettings> stored_settings;

	//if memory address isn't the same
	if (stored_settings.get() != settings) {
		//validate memory
		if (!settings->validate()) {
			return false;
		}
		//replace current settings
		//deep copy the thing
		stored_settings = std::unique_ptr<const STPSettings::STPHeightfieldSettings>(new STPSettings::STPHeightfieldSettings(*settings));
	}

	return cudaSuccess == cudaMemcpyToSymbol(HeightfieldSettings, stored_settings.get(), sizeof(STPSettings::STPHeightfieldSettings), 0ull, cudaMemcpyHostToDevice);
}

__host__ bool STPHeightfieldGenerator::generateHeightfieldCUDA(STPMapStorage& args, STPGeneratorOperation operation) const {
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
	auto isFlagged = [](STPGeneratorOperation op, STPGeneratorOperation flag) -> bool {
		return (op & flag) != 0u;
	};

	bool no_error = true;//check for error, true if all successful
	//allocating spaces for texture, storing on device
	//this is the size for a texture in one channel
	const unsigned int num_pixel = this->Noise_Settings.Dimension.x * this->Noise_Settings.Dimension.y;
	const unsigned int map_size = num_pixel * sizeof(float);
	const unsigned int map16ui_size = num_pixel * sizeof(unsigned short);
	const unsigned int freeslip_chunk_total = this->FreeSlipChunk.x * this->FreeSlipChunk.y;
	const unsigned int freeflip_pixel = freeslip_chunk_total * num_pixel;
	const unsigned int central_index = static_cast<unsigned int>(floorf(1.0f * freeslip_chunk_total / 2.0f));
	const unsigned int map_freeslip_size = freeflip_pixel * sizeof(float);
	//heightmap and normalmap ptr
	float* heightfield_d[2] = {nullptr};
	float* heightfield_freeslip_h = nullptr;
	float* heightfield_freeslip_d = nullptr;
	unsigned short* heightfield_formatted_d[2] = {nullptr};

	//Retrieve all flags
	const bool flags[4] = {
		isFlagged(operation, STPHeightfieldGenerator::HeightmapGeneration),
		isFlagged(operation, STPHeightfieldGenerator::Erosion),
		isFlagged(operation, STPHeightfieldGenerator::NormalmapGeneration),
		isFlagged(operation, STPHeightfieldGenerator::Format)
	};
	//The format flags
	const bool format_flags[2] = {
		isFlagged(args.FormatHint, STPHeightfieldGenerator::FormatHeightmap),
		isFlagged(args.FormatHint, STPHeightfieldGenerator::FormatNormalmap)
	};

	//memory allocation
	//FP32
	{
		std::unique_lock<std::mutex> lock(this->MapCache32F_lock);
		if (flags[1]) {
			heightfield_freeslip_d = this->MapCache32F_device.allocate(map_freeslip_size);
			//use the center location of the erosion memory as the central chunk heightmap
			heightfield_d[0] = heightfield_freeslip_d + central_index * num_pixel;
		}
		else {
			heightfield_d[0] = this->MapCache32F_device.allocate(map_size);
		}
		if (flags[2] || (flags[3] && format_flags[1])) {
			//if normal map formation is enabled, we need the device memory for input as well
			heightfield_d[1] = this->MapCache32F_device.allocate(map_size * 4);
		}
	}
	//INT16
	if (flags[3]) {
		{
			std::unique_lock<std::mutex> lock(this->MapCache16UI_lock);
			if (format_flags[0]) {
				heightfield_formatted_d[0] = this->MapCache16UI_device.allocate(map16ui_size);
			}
			if (format_flags[1]) {
				heightfield_formatted_d[1] = this->MapCache16UI_device.allocate(map16ui_size * 4);
			}
		}
	}
	//FP32 Host
	if (flags[1]) {
		{
			std::unique_lock<std::mutex> lock(this->MapCachePinned_lock);
			heightfield_freeslip_h = this->MapCachePinned.allocate(map_freeslip_size);
		}
	}

	//setup phase
	//creating stream so cpu thread can calculate all chunks altogether
	cudaStream_t stream;
	//we want the stream to not be blocked by default stream
	no_error &= cudaSuccess == cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	if (flags[1]) {
		//pre copy free-slip chunks to device
		//heightmap generation will overwrite the central chunk anyway
		unsigned int base = 0u;
		for (float* map : args.Heightmap32F) {
			no_error &= cudaSuccess == cudaMemcpyAsync(heightfield_freeslip_h + base, map, map_size,cudaMemcpyHostToHost, stream);
			base += num_pixel;
		}
		no_error &= cudaSuccess == cudaMemcpyAsync(heightfield_freeslip_d, heightfield_freeslip_h, map_freeslip_size, cudaMemcpyHostToDevice, stream);
	}
	
	//Flag: HeightmapGeneration
	if (flags[0]) {
		//generate a new heightmap and store it to the output later
		generateHeightmapKERNEL << <this->numBlock_Map, this->numThreadperBlock_Map, 0, stream >> > (this->simplex, heightfield_d[0],
			this->Noise_Settings.Dimension, make_float2(1.0f * this->Noise_Settings.Dimension.x / 2.0f, 1.0f * this->Noise_Settings.Dimension.y / 2.0f), args.HeightmapOffset);
	}
	else if(!flags[1]) {
		//copy heightmap from input arguments to device for later use if not generated nor going to be eroded
		no_error &= cudaSuccess == cudaMemcpyAsync(heightfield_d[0], *args.Heightmap32F.begin(), map_size, cudaMemcpyHostToDevice, stream);
	}

	//Flag: Erosion
	if (flags[1]) {
		//erode the heightmap, either from provided heightmap or generated previously
		const STPRainDrop::STPFreeSlipManager heightmap_slip(heightfield_freeslip_d, this->GlobalLocalIndex, this->FreeSlipChunk, this->Noise_Settings.Dimension);
		performErosionKERNEL << <this->numBlock_Erosion, this->numThreadperBlock_Erosion, 0, stream >> > (heightmap_slip, this->RNG_Map);
	}

	//Flag: Normalmap
	if (flags[2]) {
		//generate normalmap from heightmap
		generateNormalmapKERNEL << <this->numBlock_Map, this->numThreadperBlock_Map, 0, stream >> > (heightfield_d[0], heightfield_d[1],
			this->Noise_Settings.Dimension);
	}

	//Flag: Format - move STPImageConverter to here
	if (flags[3]) {
		if (format_flags[0]) {
			//format heightmap
			//heightmap will always be available
			//format heightmap
			floatToshortKERNEL << <this->numBlock_Map, this->numThreadperBlock_Map, 0, stream >> > (heightfield_d[0], heightfield_formatted_d[0],
				this->Noise_Settings.Dimension, 1);
		}

		if (format_flags[1]) {
			//format normalmap
			if (!flags[2]) {
				//normalmap generation was not enabled? we need to copy from input
				no_error &= cudaSuccess == cudaMemcpyAsync(heightfield_d[1], args.Normalmap32F, map16ui_size * 4, cudaMemcpyHostToDevice, stream);
			}
			//if normalmap is generated, it's already available in device memory
			floatToshortKERNEL << <this->numBlock_Map, this->numThreadperBlock_Map, 0, stream >> > (heightfield_d[1], heightfield_formatted_d[1],
				this->Noise_Settings.Dimension, 4);
		}
		
	}
	
	//Store the result accordingly
	//copy the result back to the host
	//heightmap will always be available
	if (flags[1]) {
		//copy all free-slip chunks back
		no_error &= cudaSuccess == cudaMemcpyAsync(heightfield_freeslip_h, heightfield_freeslip_d, map_freeslip_size, cudaMemcpyDeviceToHost, stream);
		unsigned int base = 0u;
		for (float* map : args.Heightmap32F) {
			no_error &= cudaSuccess == cudaMemcpyAsync(map, heightfield_freeslip_h + base, map_size, cudaMemcpyHostToHost, stream);
			base += num_pixel;
		}
	}
	else {
		no_error &= cudaSuccess == cudaMemcpyAsync(*args.Heightmap32F.begin(), heightfield_d[0], map_size, cudaMemcpyDeviceToHost, stream);
	}
	if (flags[2]) {
		//if we have normalmap generated, also copy normalmap back to host
		no_error &= cudaSuccess == cudaMemcpyAsync(args.Normalmap32F, heightfield_d[1], map_size * 4, cudaMemcpyDeviceToHost, stream);
	}
	//copy the formatted result if enabled
	if (flags[3]) {
		if (format_flags[0]) {
			//copy heightmap
			no_error &= cudaSuccess == cudaMemcpyAsync(args.Heightmap16UI, heightfield_formatted_d[0], map16ui_size, cudaMemcpyDeviceToHost, stream);
		}
		if (format_flags[1]) {
			//copy normalmap
			no_error &= cudaSuccess == cudaMemcpyAsync(args.Normalmap16UI, heightfield_formatted_d[1], map16ui_size * 4, cudaMemcpyDeviceToHost, stream);
		}
	}
	//waiting for finish
	no_error &= cudaSuccess == cudaStreamSynchronize(stream);

	//Finish up the rest, clear up when the device is ready
	//nullptr means not allocated
	{
		std::unique_lock<std::mutex> lock(this->MapCache32F_lock);
		if (flags[1]) {
			this->MapCache32F_device.deallocate(map_freeslip_size, heightfield_freeslip_d);
		}
		else {
			//hydraulic erosion uses a standalone heightmap so don't free it
			//heightfield_d[0] will never be null
			this->MapCache32F_device.deallocate(map_size, heightfield_d[0]);
		}
		if (heightfield_d[1] != nullptr) {
			this->MapCache32F_device.deallocate(map_size * 4, heightfield_d[1]);
		}
	}
	if (flags[3]) {
		{
			std::unique_lock<std::mutex> lock(this->MapCache16UI_lock);
			if (format_flags[0]) {
				this->MapCache16UI_device.deallocate(map16ui_size, heightfield_formatted_d[0]);
			}
			if (format_flags[1]) {
				this->MapCache16UI_device.deallocate(map16ui_size * 4, heightfield_formatted_d[1]);
			}
		}
	}
	if (flags[1]) {
		{
			std::unique_lock<std::mutex> lock(this->MapCachePinned_lock);
			this->MapCachePinned.deallocate(map_freeslip_size, heightfield_freeslip_h);
		}
	}
	no_error &= cudaSuccess == cudaStreamDestroy(stream);

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

__host__ bool STPHeightfieldGenerator::setLocalGlobalIndexCUDA() {
	//TODO: Don't generate the table when FreeSlipChunk.xy are both 1, and in STPRainDrop don't use the table
	const uint2& dimension = this->Noise_Settings.Dimension;
	const uint2& range = this->FreeSlipChunk;

	const uint2 global_dimension = make_uint2(range.x * dimension.x, range.y * dimension.y);
	//launch parameters
	dim3 numBlock_Index = dim3(global_dimension.x / this->numThreadperBlock_Map.x, global_dimension.y / this->numThreadperBlock_Map.y);
	bool no_error = true;

	//make sure all previous takes are finished
	no_error &= cudaSuccess == cudaDeviceSynchronize();
	//allocation
	if (this->GlobalLocalIndex != nullptr) {
		no_error &= cudaSuccess == cudaFree(this->GlobalLocalIndex);
	}
	no_error &= cudaSuccess == cudaMalloc(&this->GlobalLocalIndex, sizeof(unsigned int) * global_dimension.x * global_dimension.y);

	//compute
	initGlobalLocalIndexKERNEL << <numBlock_Index, this->numThreadperBlock_Map >> > (this->GlobalLocalIndex, global_dimension.x, range, dimension);
	no_error &= cudaSuccess == cudaDeviceSynchronize();

	return no_error;
}

__device__ __inline__ float3 normalize3DKERNEL(float3 vec3) {
	const float length = sqrtf(powf(vec3.x, 2) + powf(vec3.y, 2) + powf(vec3.z, 2));
	return make_float3(fdividef(vec3.x, length), fdividef(vec3.y, length), fdividef(vec3.z, length));
}

__device__ __inline__ float InvlerpKERNEL(float minVal, float maxVal, float value) {
	//lerp the noiseheight to [0,1]
	return __saturatef(fdividef(value - minVal, maxVal - minVal));
}

__device__ __forceinline__ int clamp(int val, int lower, int upper) {
	return max(lower, min(val, upper));
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
	SuperTerrainPlus::STPSettings::STPRainDropSettings* const settings = (SuperTerrainPlus::STPSettings::STPRainDropSettings* const)(reinterpret_cast<const SuperTerrainPlus::STPSettings::STPHeightfieldSettings* const>(HeightfieldSettings));

	//current working index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	//generating random location
	//first we generate the number (0.0f, 1.0f]
	float2 initPos = make_float2(curand_uniform(&rng[index]), curand_uniform(&rng[index]));
	//convert to (erode radius + base, dimension - erode radius - 1]
	//range: dimension - 2 * erosion radius - 1
	//Generate the raindrop at the central chunk only
	//TODO precompute constants outside of device
	initPos.x *= heightmap_storage.Dimension.x - 2.0f * settings->getErosionBrushRadius() - 1.0f;
	initPos.x += settings->getErosionBrushRadius() + (floorf(heightmap_storage.FreeSlipChunk.x / 2) * heightmap_storage.Dimension.x);
	initPos.y *= heightmap_storage.Dimension.y - 2.0f * settings->getErosionBrushRadius() - 1.0f;
	initPos.y += settings->getErosionBrushRadius() + (floorf(heightmap_storage.FreeSlipChunk.y / 2) * heightmap_storage.Dimension.y);

	//spawn in the raindrop
	STPRainDrop droplet(initPos, settings->initWaterVolume, settings->initSpeed);
	//usually each droplet only does that once, rarely go beyond twice.
	//Just adding in case...
	droplet.Erode(settings, heightmap_storage);
}

__global__ void generateNormalmapKERNEL(float* const heightmap, float* normal_storage, uint2 dimension) {
	//convert constant memory to usable class
	SuperTerrainPlus::STPSettings::STPHeightfieldSettings* const settings = reinterpret_cast<SuperTerrainPlus::STPSettings::STPHeightfieldSettings* const>(HeightfieldSettings);

	//the current working pixel
	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y;

	//load the cells from heightmap, remember the height map only contains one color channel
	//using Sobel fitering
	float cell[8];
	cell[0] = heightmap[clamp((x - 1), 0, dimension.x - 1) + clamp((y - 1), 0, dimension.y - 1) * dimension.x];
	cell[1] = heightmap[x + clamp((y - 1), 0, dimension.y - 1) * dimension.x];
	cell[2] = heightmap[clamp((x + 1), 0, dimension.x - 1) + clamp((y - 1), 0, dimension.y - 1) * dimension.x];
	cell[3] = heightmap[clamp((x - 1), 0, dimension.x - 1) + y * dimension.x];
	cell[4] = heightmap[clamp((x + 1), 0, dimension.x - 1) + y * dimension.x];
	cell[5] = heightmap[clamp((x - 1), 0, dimension.x - 1) + clamp((y + 1), 0, dimension.y - 1) * dimension.x];
	cell[6] = heightmap[x + clamp((y + 1), 0, dimension.y - 1) * dimension.x];
	cell[7] = heightmap[clamp((x + 1), 0, dimension.x - 1) + clamp((y + 1), 0, dimension.y - 1) * dimension.x];
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
	normal_storage[(x + y * dimension.x) * 4] = normal.x;//R
	normal_storage[(x + y * dimension.x) * 4 + 1] = normal.y;//G
	normal_storage[(x + y * dimension.x) * 4 + 2] = normal.z;//B
	normal_storage[(x + y * dimension.x) * 4 + 3] = 1.0f;//A
	
	return;
}

__global__ void floatToshortKERNEL(const float* const input, unsigned short* output, uint2 dimension, unsigned int channel) {
	//current working pixel
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y,
		index = x + y * dimension.x;

	//loop through all channels and output
	for (int i = 0; i < channel; i++) {
		output[index * channel + i] = static_cast<unsigned short>(input[index * channel + i] * 65535u);
	}
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
