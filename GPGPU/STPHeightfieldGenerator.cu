#pragma once
#include "STPHeightfieldGenerator.cuh"

__constant__ unsigned char HeightfieldSettings[sizeof(SuperTerrainPlus::STPSettings::STPHeightfieldSettings)];

using namespace SuperTerrainPlus::STPCompute;

/**
 * @brief Kernel launch and util functions
*/
namespace STPKernelLauncher {

	/**
	 * @brief Find the unit vector of the input vector
	 * @param vec3 - Vector input
	 * @return Unit vector of the input
	*/
	__device__ float3 normalize3DKERNEL(float3);

	/**
	 * @brief Performing inverse linear interpolation for each value on the heightmap to scale it within [0,1] using CUDA kernel
	 * @param minVal The mininmum value that can apperar in this height map
	 * @param maxVal The maximum value that can apperar in this height map
	 * @param value The input value
	 * @return The interpolated value
	*/
	__device__ float InvlerpKERNEL(float, float, float);

	/**
	 * @brief Clamp the input value with the range
	 * @param val The clamping value
	 * @param lower The lowest possible value
	 * @param upper The highest possible value
	 * @return val if [lower, upper], lower if val < lower, upper if val > upper
	*/
	__device__ int clamp(int, int, int);

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
	 * @param height_storage Heightmap that is going to erode with raindrop
	 * @param dimension The size of all maps, they must be the same
	 * @param rng The random number generator map sequence, independent for each rain drop
	*/
	__global__ void performErosionKERNEL(float*, uint2, STPHeightfieldGenerator::curandRNG*);

	/**
	* @brief Generate the normal map for the height map within kernel
	* @param heightmap - contains the height map that will be wused to generate the normal
	* @param normal_storage - normal map, will be used to store the output of the normal map
	* @param dimension - The width and height of both map
	* @return True if the normal map is successully generated without errors
	*/
	__global__ void generateNormalmapKERNEL(float* const, float*, uint2);

}

__host__ float* STPHeightfieldGenerator::STPHeightfieldAllocator::allocate(size_t count) {
	float* mem = nullptr;
	cudaMalloc(&mem, sizeof(float) * count);
	return mem;
}

__host__ void STPHeightfieldGenerator::STPHeightfieldAllocator::deallocate(size_t count, float* ptr) {
	cudaFree(ptr);
}

__host__ STPHeightfieldGenerator::STPHeightfieldGenerator(STPSettings::STPSimplexNoiseSettings* const noise_settings) : simplex_h(noise_settings), Noise_Settings(*noise_settings) {
	//allocating space
	cudaMalloc(&this->simplex, sizeof(STPSimplexNoise));
	//copy data
	cudaMemcpy(this->simplex, &simplex_h, sizeof(STPSimplexNoise), cudaMemcpyHostToDevice);

	//kernel parameters
	this->numThreadperBlock_Map = dim3(32, 32);
	this->numBlock_Map = dim3(noise_settings->Dimension.x / numThreadperBlock_Map.x, noise_settings->Dimension.y / numThreadperBlock_Map.y);
	this->numThreadperBlock_Erosion = 1024;
	this->numBlock_Erosion = 0;//This will be set after user call the setErosionIterationCUDA() method
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
}

__host__ bool STPHeightfieldGenerator::useSettings(const STPSettings::STPHeightfieldSettings* const settings) {
	//keep a local copy of the setting so device can have access to the pointer inside the class
	static const STPSettings::STPHeightfieldSettings* stored_settings = nullptr;

	if (settings == nullptr) {
		//clear memory
		delete stored_settings;
		return true;
	}
	if (stored_settings != settings) {
		//validate memory
		if (!settings->validate()) {
			return false;
		}
		//replace current settings
		if (stored_settings != nullptr) {
			delete stored_settings;
		}
		//deep copy the thing
		stored_settings = new STPSettings::STPHeightfieldSettings(*settings);
	}

	return cudaSuccess == cudaMemcpyToSymbol(HeightfieldSettings, stored_settings, sizeof(STPSettings::STPHeightfieldSettings), 0ull, cudaMemcpyHostToDevice);
}

__host__ bool STPHeightfieldGenerator::generateHeightfieldCUDA(float* heightmap, float* normalmap, float3 offset) const {
	//check the availiability of the engine
	if (this->RNG_Map == nullptr) {
		return false;
	}
	//check the availability of biome dictionary
	/*if (this->BiomeDictionary == nullptr) {
		return false;
	}*/
	
	//allocating spaces for texture, storing on device
	//this is the size for a texture in one channel
	const int num_pixel = this->Noise_Settings.Dimension.x * this->Noise_Settings.Dimension.y;
	const int map_size = num_pixel * sizeof(float);
	float* heightfield_d[2] = {nullptr};//heightmap and normalmap

	bool no_error = true;//check for error, true if all successful
	//regarding the size of the heightfields, heightmap, streammap and poolmap are all having R32F format, while normalmap uses RGBA32F
	//so there are 7 channels in total
	{
		std::unique_lock<std::mutex> lock(this->memorypool_lock);
		heightfield_d[0] = this->MapCache_device.allocate(map_size);
		heightfield_d[1] = this->MapCache_device.allocate(map_size * 4);
	}

	//creating stream so cpu thread can calculate all chunks altogether
	cudaStream_t stream;
	//we want the stream to not be blocked by default stream
	no_error &= cudaSuccess == cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

	//calculate heightmap
	STPKernelLauncher::generateHeightmapKERNEL << <this->numBlock_Map, this->numThreadperBlock_Map, 0, stream >> > (this->simplex, heightfield_d[0],
		this->Noise_Settings.Dimension, make_float2(1.0f * this->Noise_Settings.Dimension.x / 2.0f, 1.0f * this->Noise_Settings.Dimension.y / 2.0f), offset);
	//performing erosion
	STPKernelLauncher::performErosionKERNEL << <this->numBlock_Erosion, this->numThreadperBlock_Erosion, 0, stream >> > (heightfield_d[0], 
		this->Noise_Settings.Dimension, this->RNG_Map);
	//calculating normalmap
	STPKernelLauncher::generateNormalmapKERNEL << <this->numBlock_Map, this->numThreadperBlock_Map, 0, stream >> > (heightfield_d[0], heightfield_d[1],
		this->Noise_Settings.Dimension);
	
	//copy the result back to the host
	no_error &= cudaSuccess == cudaMemcpyAsync(heightmap, heightfield_d[0], map_size, cudaMemcpyDeviceToHost, stream);
	no_error &= cudaSuccess == cudaMemcpyAsync(normalmap, heightfield_d[1], map_size * 4, cudaMemcpyDeviceToHost, stream);

	//block the host thread 
	no_error &= cudaSuccess == cudaStreamSynchronize(stream);
	//clear up when the device is ready
	{
		std::unique_lock<std::mutex> lock(this->memorypool_lock);
		this->MapCache_device.deallocate(map_size, heightfield_d[0]);
		this->MapCache_device.deallocate(map_size * 4, heightfield_d[1]);
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
	STPKernelLauncher::curandInitKERNEL<<<this->numBlock_Erosion, this->numThreadperBlock_Erosion>>>(this->RNG_Map, this->Noise_Settings.Seed);
	no_error &= cudaSuccess == cudaDeviceSynchronize();
	//leave the result on device, and update the raindrop count
	this->NumRaindrop = raindrop_count;

	return no_error;
}

__host__ unsigned int STPHeightfieldGenerator::getErosionIteration() const {
	return this->NumRaindrop;
}

__device__ float3 STPKernelLauncher::normalize3DKERNEL(float3 vec3) {
	const float length = sqrtf(powf(vec3.x, 2) + powf(vec3.y, 2) + powf(vec3.z, 2));
	return make_float3(fdividef(vec3.x, length), fdividef(vec3.y, length), fdividef(vec3.z, length));
}

__device__ float STPKernelLauncher::InvlerpKERNEL(float minVal, float maxVal, float value) {
	//lerp the noiseheight to [0,1]
	return __saturatef(fdividef(value - minVal, maxVal - minVal));
}

__device__ int STPKernelLauncher::clamp(int val, int lower, int upper) {
	return max(lower, min(val, upper));
}

__global__ void STPKernelLauncher::curandInitKERNEL(STPHeightfieldGenerator::curandRNG* rng, unsigned long long seed) {
	//current working index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	//the same seed but we are looking for different sequence
	curand_init(seed, static_cast<unsigned long long>(index), 0, &rng[index]);
}

__global__ void STPKernelLauncher::generateHeightmapKERNEL(STPSimplexNoise* const noise_fun, float* height_storage,
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

__global__ void STPKernelLauncher::performErosionKERNEL(float* height_storage, uint2 dimension, STPHeightfieldGenerator::curandRNG* rng) {
	//convert constant memory to usable class
	SuperTerrainPlus::STPSettings::STPRainDropSettings* const settings = 
		reinterpret_cast<SuperTerrainPlus::STPSettings::STPHeightfieldSettings* const>(HeightfieldSettings);

	//current working index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	//generating random location
	//first we generate the number (0.0f, 1.0f]
	float2 initPos = make_float2(curand_uniform(&rng[index]), curand_uniform(&rng[index]));
	//convert to (erode radius, dimension - erode radius - 1]
	//range: dimension - 2 * erosion radius - 1
	initPos.x *= dimension.x - 2.0f * settings->getErosionBrushRadius() - 1.0f;
	initPos.x += settings->getErosionBrushRadius();
	initPos.y *= dimension.y - 2.0f * settings->getErosionBrushRadius() - 1.0f;
	initPos.y += settings->getErosionBrushRadius();

	//spawn in the raindrop
	STPRainDrop droplet(initPos, settings->initWaterVolume, settings->initSpeed);
	//usually each droplet only does that once, rarely go beyond twice.
	//Just adding in case...
	droplet.Erode(settings, dimension, height_storage);
}

__global__ void STPKernelLauncher::generateNormalmapKERNEL(float* const heightmap, float* normal_storage, uint2 dimension) {
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