#pragma once
#include "STPHeightfieldGenerator.cuh"

using namespace SuperTerrainPlus::STPCompute;

__host__ STPHeightfieldGenerator::STPHeightfieldGenerator(STPSettings::STPHeightfieldSettings* const settings,  STPSettings::STPSimplexNoiseSettings* const noise_settings) {
	//Init simplex noise generator and store it inside the device
	STPSimplexNoise simplex_h(noise_settings);
	//allocating space
	cudaMalloc(&this->simplex, sizeof(STPSimplexNoise));
	//copy data
	cudaMemcpy(this->simplex, &simplex_h, sizeof(STPSimplexNoise), cudaMemcpyHostToDevice);

	//copy the parameters to device
	cudaMalloc(&this->Settings, sizeof(STPSettings::STPHeightfieldSettings));
	//this is a deep move on STPHeightfieldSettings
	STPSettings::STPHeightfieldSettings settings_cpy(std::move(*settings));
	cudaMemcpy(this->Settings, &settings_cpy, sizeof(STPSettings::STPHeightfieldSettings), cudaMemcpyHostToDevice);

	//kernel parameters
	this->numThreadperBlock_Map = dim3(32, 32);
	this->numBlock_Map = dim3(noise_settings->Dimension.x / numThreadperBlock_Map.x, noise_settings->Dimension.y / numThreadperBlock_Map.y);
	this->numThreadperBlock_Erosion = 1024;
	this->numBlock_Erosion = 0;//This will be set after user call the setErosionIterationCUDA() method
	this->Noise_Settings = new STPSettings::STPSimplexNoiseSettings(*noise_settings);
}

__host__ STPHeightfieldGenerator::~STPHeightfieldGenerator() {
	cudaFree(this->simplex);
	cudaFree(this->Settings);
	//check if the rng has been init
	if (this->RNG_Map != nullptr) {
		cudaFree(this->RNG_Map);
	}
	delete this->Noise_Settings;
}

__host__ bool STPHeightfieldGenerator::generateHeightfieldCUDA(float* heightmap, float* normalmap, float3 offset) {
	//check the availiability of the engine
	if (this->Settings == nullptr || this->RNG_Map == nullptr) {
		return false;
	}
	
	//allocating spaces for texture, storing on device
	//this is the size for a texture in one channel
	const int num_pixel = this->Noise_Settings->Dimension.x * this->Noise_Settings->Dimension.y;
	const int map_size = num_pixel * sizeof(float);
	float* heightfield_d[2] = {nullptr};//heightmap and normalmap

	bool no_error = true;//check for error, true if all successful
	//regarding the size of the heightfields, heightmap, streammap and poolmap are all having R32F format, while normalmap uses RGBA32F
	//so there are 7 channels in total
	no_error &= cudaSuccess == cudaMalloc(&(heightfield_d[0]), map_size);
	no_error &= cudaSuccess == cudaMalloc(&(heightfield_d[1]), map_size * 4);

	//creating stream so cpu thread can calculate all chunks altogether
	cudaStream_t stream;
	//we want the stream to not be blocked by default stream
	no_error &= cudaSuccess == cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

	//calculate heightmap
	STPKernelLauncher::generateHeightmapKERNEL << <this->numBlock_Map, this->numThreadperBlock_Map, 0, stream >> > (this->simplex, heightfield_d[0],
		this->Noise_Settings->Dimension, make_float2(1.0f * this->Noise_Settings->Dimension.x / 2.0f, 1.0f * this->Noise_Settings->Dimension.y / 2.0f), this->Settings, offset);
	//performing erosion
	STPKernelLauncher::performErosionKERNEL << <this->numBlock_Erosion, this->numThreadperBlock_Erosion, 0, stream >> > (heightfield_d[0], 
		this->Noise_Settings->Dimension, this->RNG_Map, dynamic_cast<STPSettings::STPRainDropSettings*>(this->Settings));
	//calculating normalmap
	STPKernelLauncher::generateNormalmapKERNEL << <this->numBlock_Map, this->numThreadperBlock_Map, 0, stream >> > (this->simplex, heightfield_d[0], heightfield_d[1],
		this->Noise_Settings->Dimension, this->Settings);
	
	//copy the result back to the host
	no_error &= cudaSuccess == cudaMemcpyAsync(heightmap, heightfield_d[0], map_size, cudaMemcpyDeviceToHost, stream);
	no_error &= cudaSuccess == cudaMemcpyAsync(normalmap, heightfield_d[1], map_size * 4, cudaMemcpyDeviceToHost, stream);

	//block the host thread 
	no_error &= cudaSuccess == cudaStreamSynchronize(stream);
	//clear up when the device is ready
	no_error &= cudaSuccess == cudaFree(heightfield_d[0]);
	no_error &= cudaSuccess == cudaFree(heightfield_d[1]);
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
	STPKernelLauncher::curandInitKERNEL<<<this->numBlock_Erosion, this->numThreadperBlock_Erosion>>>(this->RNG_Map, this->Noise_Settings->Seed);
	no_error &= cudaSuccess == cudaDeviceSynchronize();
	//leave the result on device, and update the raindrop count
	this->NumRaindrop = raindrop_count;

	return no_error;
}

__host__ int STPHeightfieldGenerator::getErosionIteration() {
	return this->NumRaindrop;
}

__host__ bool STPHeightfieldGenerator::setHeightfieldParameter(const STPSettings::STPHeightfieldSettings& settings) {
	return cudaMemcpy(this->Settings, &settings, sizeof(STPSettings::STPHeightfieldSettings), cudaMemcpyHostToDevice) == cudaSuccess;
}

__host__ SuperTerrainPlus::STPSettings::STPHeightfieldSettings* STPHeightfieldGenerator::getHeightfieldParameter() {
	STPSettings::STPHeightfieldSettings* para = nullptr;
	//pinned memory
	cudaMallocHost(&para, sizeof(STPSettings::STPHeightfieldSettings*));
	cudaMemcpy(para, this->Settings, sizeof(STPSettings::STPHeightfieldSettings*), cudaMemcpyDeviceToHost);
	return para;
}

__host__ void STPHeightfieldGenerator::freeHeightfieldParameter(STPSettings::STPHeightfieldSettings* const memory) {
	cudaFreeHost(memory);
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
	uint2 dimension, float2 half_dimension, STPSettings::STPHeightfieldSettings* const settings, float3 offset) {
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

__global__ void STPKernelLauncher::performErosionKERNEL(float* height_storage, uint2 dimension, STPHeightfieldGenerator::curandRNG* rng, STPSettings::STPRainDropSettings* const settings) {
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

__global__ void STPKernelLauncher::generateNormalmapKERNEL(STPSimplexNoise* const noise_fun,
	float* const heightmap, float* normal_storage, uint2 dimension, STPSettings::STPHeightfieldSettings* const settings) {
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