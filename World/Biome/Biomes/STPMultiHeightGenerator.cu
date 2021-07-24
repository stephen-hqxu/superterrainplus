#pragma once
#include "STPMultiHeightGenerator.cuh"

//CUDA
#include <device_launch_parameters.h>
//Error Handling
#include "../../../GPGPU/STPDeviceErrorHandler.h"

using namespace SuperTerrainPlus::STPCompute;
using namespace STPDemo;

static __constant__ unsigned char BiomeSetting_raw[sizeof(STPBiomeSettings)]; 
static __constant__ uint2 Dimension[1];
static __constant__ float2 HalfDimension[1];
static __device__ unsigned char Simplex_raw[sizeof(STPSimplexNoise)];

/**
 * @brief Performing inverse linear interpolation for each value on the heightmap to scale it within [0,1] using CUDA kernel
 * @param minVal The mininmum value that can apperar in this height map
 * @param maxVal The maximum value that can apperar in this height map
 * @param value The input value
 * @return The interpolated value
*/
__device__ __inline__ float InvlerpKERNEL(float, float, float);

/**
 * @brief Generate our epic height map using simplex noise function within the CUDA kernel
 * @param height_storage - The pointer to a location where the heightmap will be stored
 * @param offset - Controlling the offset on x, y directions
*/
__global__ void generateMultiBiomeHeightmap(float*, float2);

__host__ void STPMultiHeightGenerator::initGenerator(const STPBiomeSettings* biome_settings, const SuperTerrainPlus::STPCompute::STPSimplexNoise* simplex_generator, uint2 dimension) {
	//copy dimension
	STPcudaCheckErr(cudaMemcpyToSymbol(Dimension, &dimension, sizeof(uint2), 0ull, cudaMemcpyHostToDevice));
	const float2 half_dimension = make_float2(dimension.x / 2.0f , dimension.y / 2.0f);
	STPcudaCheckErr(cudaMemcpyToSymbol(HalfDimension, &half_dimension, sizeof(float2), 0ull, cudaMemcpyHostToDevice));
	//copy biome settings
	STPcudaCheckErr(cudaMemcpyToSymbol(BiomeSetting_raw, biome_settings, sizeof(STPBiomeSettings), 0ull, cudaMemcpyHostToDevice));
	//copy noise gen
	STPcudaCheckErr(cudaMemcpyToSymbol(Simplex_raw, simplex_generator, sizeof(STPSimplexNoise), 0ull, cudaMemcpyHostToDevice));
}

__host__ void STPMultiHeightGenerator::generateHeightmap(float* heightmap, uint2 dimension, float2 offset, cudaStream_t stream) {
	int Mingridsize, blocksize;
	dim3 Dimgridsize, Dimblocksize;
	//smart launch config
	STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &generateMultiBiomeHeightmap));
		Dimblocksize = dim3(32, blocksize / 32);
		Dimgridsize = dim3((dimension.x + Dimblocksize.x - 1) / Dimblocksize.x, (dimension.y + Dimblocksize.y - 1) / Dimblocksize.y);
	//launch
	generateMultiBiomeHeightmap << <Dimgridsize, Dimblocksize, 0, stream >> > (heightmap, offset);
}

//--------------------- Definition --------------------------

__device__ __inline__ float InvlerpKERNEL(float minVal, float maxVal, float value) {
	//lerp the noiseheight to [0,1]
	return __saturatef(fdividef(value - minVal, maxVal - minVal));
}

//TODO: Make sure it works as before, and then generate multi-biome heightmap
__global__ void generateMultiBiomeHeightmap(float* height_storage, float2 offset) {
	//the current working pixel
	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= Dimension->x || y >= Dimension->y) {
		return;
	}
	//convert
	const STPBiomeSettings* BiomeSetting = reinterpret_cast<STPBiomeSettings*>(BiomeSetting_raw);
	const STPSimplexNoise* Simplex = reinterpret_cast<STPSimplexNoise*>(Simplex_raw);

	float amplitude = 1.0f, frequency = 1.0f, noiseheight = 0.0f;
	float min = 0.0f, max = 0.0f;//The min and max indicates the range of the multi-phased simplex function, not the range of the output texture
	//multiple phases of noise
	for (int i = 0; i < BiomeSetting->Octave; i++) {
		float sampleX = ((1.0 * x - HalfDimension->x) + offset.x) / BiomeSetting->Scale * frequency, //subtract the half width and height can make the scaling focus at the center
			sampleY = ((1.0 * y - HalfDimension->y) + offset.y) / BiomeSetting->Scale * frequency;//since the y is inverted we want to filp it over
		noiseheight += Simplex->simplex2D(sampleX, sampleY) * amplitude;

		//calculate the min and max
		min -= 1.0f * amplitude;
		max += 1.0f * amplitude;
		//scale the parameters
		amplitude *= BiomeSetting->Persistence;
		frequency *= BiomeSetting->Lacunarity;
	}

	//interpolate and clamp the value within [0,1], was [min,max]
	noiseheight = InvlerpKERNEL(min, max, noiseheight);
	//finally, output the texture
	height_storage[x + y * Dimension->x] = noiseheight;//we have only allocated R32F format;

	return;
}