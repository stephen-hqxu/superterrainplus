#include "./Script/STPCommonGenerator.cuh"

//SuperAlgorithm+ Device library
#include <STPSimplexNoise.cuh>
#include <STPKernelMath.cuh>
#include <STPSingleHistogramWrapper.cuh>

//Biome parameters
#include <STPBiomeProperty>

using namespace SuperTerrainPlus::STPCompute;
using SuperTerrainPlus::STPDiversity::Sample;

__constant__ STPDemo::STPBiomeProperty BiomeTable[2];
__constant__ STPPermutation Permutation[1];

/**
 * @brief Use pure simplex noise to sample a point fractally.
 * @param coord The coordinate on the texture
 * @param parameter The simplex parameter for this biome
 * @param offset The simplex noise offset in global coordinate
 * @return The normalised noise value.
*/
__device__ float sampleSimplexNoise(uint2, const STPDemo::STPBiomeProperty&, float2);

//--------------------- Definition --------------------------

using namespace STPCommonGenerator;

/**
 * @brief Generate our epic height map using simplex noise function within the CUDA kernel
 * @param height_storage - The pointer to a location where the heightmap will be stored
 * @param biomemap_histogram - The biomemap histogram to decide the weight of each biome in a pixel
 * @param offset - Controlling the offset on x, y directions
*/
__global__ void generateMultiBiomeHeightmap(float* height_storage, STPSingleHistogram biomemap_histogram, float2 offset) {
	//the current thread index, starting from top-left corner
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= Dimension->x || y >= Dimension->y) {
		return;
	}
	//current working pixel
	const unsigned int index = x + y * Dimension->x;
	STPSingleHistogramWrapper interpolator(biomemap_histogram);
	
	//grab the current biome setting
	//we need to always make sure current biome can be referenced by the biomeID given in biome table
	float height = 0.0f;
	interpolator(index, [&height, &offset, x, y](Sample biomeID, float weight) {
		const STPDemo::STPBiomeProperty& current_biome = BiomeTable[biomeID];
		height += weight * sampleSimplexNoise(make_uint2(x, y), current_biome, offset);
	});
	
	//generate simplex noise terrain
	//finally, output the texture
	height_storage[index] = height;//we have only allocated R32F format;
	
}

__device__ float sampleSimplexNoise(uint2 coord, const STPDemo::STPBiomeProperty& parameter, float2 offset) {
	//get simplex noise generator
	const STPSimplexNoise Simplex(*Permutation);

	float amplitude = 1.0f, frequency = 1.0f, noiseheight = 0.0f;
	float min = 0.0f, max = 0.0f;//The min and max indicates the range of the multi-phased simplex function, not the range of the output texture
	//multiple phases of noise
	#pragma unroll
	for (int i = 0; i < parameter.Octave; i++) {
		float sampleX = ((1.0 * coord.x - HalfDimension->x) + offset.x) / parameter.Scale * frequency, //subtract the half width and height can make the scaling focus at the center
			sampleY = ((1.0 * coord.y - HalfDimension->y) + offset.y) / parameter.Scale * frequency;//since the y is inverted we want to filp it over
		noiseheight += Simplex.simplex2D(sampleX, sampleY) * amplitude;

		//calculate the min and max
		min -= 1.0f * amplitude;
		max += 1.0f * amplitude;
		//scale the parameters
		amplitude *= parameter.Persistence;
		frequency *= parameter.Lacunarity;
	}
	
	//interpolate and clamp the value within [0,1], was [min,max]
	noiseheight = STPKernelMath::Invlerp(min, max, noiseheight);
	//scale the noise
	noiseheight *= parameter.Variation;
	noiseheight += parameter.Depth;
	
	return noiseheight;
}