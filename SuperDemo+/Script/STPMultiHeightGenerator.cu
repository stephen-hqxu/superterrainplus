#include "./Script/STPCommonGenerator.cuh"

//SuperAlgorithm+ Device library
#include <STPSingleHistogramWrapper.cuh>

//Biome parameters
#include <STPBiomeProperty>

using namespace SuperTerrainPlus::STPAlgorithm;
using SuperTerrainPlus::STPDiversity::Sample;

__constant__ STPDemo::STPBiomeProperty BiomeTable[2];

/**
 * @brief Use pure simplex noise to sample a point with fractal.
 * @param coord The coordinate on the texture
 * @param parameter The simplex parameter for this biome
 * @param offset The simplex noise offset in global coordinate
 * @return The normalised noise value.
*/
__device__ static float sampleSimplexNoise(uint2, const STPDemo::STPBiomeProperty&, float2);

//--------------------- Definition --------------------------

using namespace STPCommonGenerator;

/**
 * @brief Generate our epic height map using simplex noise function within the CUDA kernel
 * @param height_storage - The pointer to a location where the heightmap will be stored
 * @param biomemap_histogram - The biomemap histogram to decide the weight of each biome in a pixel
 * @param offset - Controlling the offset on x, y directions
*/
__global__ void generateMultiBiomeHeightmap(float* const height_storage, const STPSingleHistogram biomemap_histogram, const float2 offset) {
	//the current thread index, starting from top-left corner
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= Dimension->x || y >= Dimension->y) {
		return;
	}
	//current working pixel
	const unsigned int index = x + y * Dimension->x;
	
	//grab the current biome setting
	//we need to always make sure current biome can be referenced by the biomeID given in biome table
	float height = 0.0f;
	STPSingleHistogramWrapper::iterate(biomemap_histogram, index, [&height, &offset, x, y](Sample biomeID, float weight) {
		const STPDemo::STPBiomeProperty& current_biome = BiomeTable[biomeID];
		height += weight * sampleSimplexNoise(make_uint2(x, y), current_biome, offset);
	});
	
	//generate simplex noise terrain
	//finally, output the texture
	height_storage[index] = height;//we have only allocated R32F format;
	
}

__device__ float sampleSimplexNoise(const uint2 coord, const STPDemo::STPBiomeProperty& parameter, const float2 offset) {
	const auto [scale, octave, pers, lacu, depth, variation] = parameter;

	//prepare for the fractal generator
	STPSimplexNoise::STPFractalSimplexInformation fractal_desc = { };
	fractal_desc.Persistence = pers;
	fractal_desc.Lacunarity = lacu;
	fractal_desc.Octave = octave;
	fractal_desc.HalfDimension = *HalfDimension;
	fractal_desc.Offset = offset;
	fractal_desc.Scale = scale;
	
	//scale the output noise
	return STPSimplexNoise::simplex2DFractal(*Permutation, 1.0f * coord.x, 1.0f * coord.y, fractal_desc)
		* variation + depth;
}