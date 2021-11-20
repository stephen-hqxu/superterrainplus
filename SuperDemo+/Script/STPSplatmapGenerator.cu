#ifndef __CUDACC_RTC__
#error __FILE__ can only be compiled in NVRTC
#endif//__CUDACC_RTC__

//SuperAlgorithm+ Device library
#include <STPTextureSplatRuleWrapper.cuh>

using namespace SuperTerrainPlus::STPCompute;
using SuperTerrainPlus::STPDiversity::Sample;

namespace STPTI = SuperTerrainPlus::STPDiversity::STPTextureInformation;

//This is the dimension of map of one chunk
__constant__ uint2 MapDimension[1];
//This is the dimension of map in the entire rendered chunk
__constant__ uint2 TotalBufferDimension[1];

__constant__ STPTI::STPSplatRuleDatabase SplatDatabase[1];
__constant__ float GradientBias[1];

//A simple 2x2 Sobel kernel
constexpr static unsigned int GradientSize = 5u;
constexpr static int2 GradientKernel[GradientSize] = {
	int2{ 0, -1 },//top, 0
	int2{ -1, 0 },//left, 1
	int2{ 0, 0 },//centre, 2
	int2{ 1, 0 },//right, 3
	int2{ 0, 1 }//bottom, 4
};

//--------------------- Definition --------------------------

/**
 * @brief Launch kernel to start splatmap generation.
 * All texture objects are non-layered 2D.
 * @param biomemap_tex The biomemap texture input
 * @param heightmap_tex The heightmap texture input
 * @param splatmap_surf The splatmap surface output
 * @param splat_info Information about the generation.
*/
__global__ void generateTextureSplatmap
	(cudaTextureObject_t biomemap_tex, cudaTextureObject_t heightmap_tex, cudaSurfaceObject_t splatmap_surf, STPTI::STPSplatGeneratorInformation splat_info) {
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y,
		//block is in 2D, so threadIdx.z is always 0 and blockDim.z is always 1
		z = blockIdx.z;
	if (x >= Dimension->x || y >= Dimension->y || z >= splat_info.LocalCount) {
		return;
	}
	//working pixel
	//we need to convert z-coord of thread to chunk local ID
	const STPTI::STPSplatGeneratorInformation::STPLocalChunkInformation& local_info = splat_info.RequestingLocalInfo[z];

	//coordinates are normalised
	const uint2 SamplingPosition = make_uint2(
		x + MapDimension->x * local_info.LocalChunkCoordinateX,
		y + MapDimension->y * local_info.LocalChunkCoordinateY
	);
	const float2 unitUV = make_float2(
		1.0f / (1.0f * TotalBufferDimension->x),
		1.0f / (1.0f * TotalBufferDimension->y)
	);
	const float2 UV = make_float2(
		1.0f * SamplingPosition.x * unitUV.x,
		1.0f * SamplingPosition.y * unitUV.y
	);

	float cell[GradientSize];
	//calculate heightmap gradient
	for (unsigned int = 0u; i < GradientSize; i++) {
		const int2& currentKernel = GradientKernel[i];
		const float2 offsetUV = make_float2(
			unitUV.x * currentKernel.x,
			unitUV.y * currentKernel.y
		);
		const float2 SamplingUV = make_float2(
			UV.x + offsetUV.x,
			UV.y + offsetUV.y
		);
		//sample this heightmap value
		cell[i] = tex2D<float>(heightmap_tex, SamplingUV.x, SamplingUV.y);
	}

	//calculate gradient using a very simple 2x2 filter, ranged [-1,1]
	const float gradient[3] = {
		((cell[1] - cell[2]) + (cell[2] - cell[3])) * 0.5f,
		((cell[4] - cell[2]) + (cell[2] - cell[0])) * 0.5f,
		*GradientBias
	};
	const float slopFactor = 1.0f - (*GradientBias * rnormf(3, gradient));

	//get information about the current position
	const Sample biome = tex2D<Sample>(biomemap_tex, UV.x, UV.y);
	const float height = tex2D<float>(heightmap_tex, UV.x, UV.y);
	const STPTextureSplatRuleWrapper splatWrapper(SplatDatabase[0]);
	//get regions, we define gradient region outweights altitude region if they overlap
	//TODO: later we can add some simplex noise to the slopFactor and height value
	unsigned int region = splatWrapper.gradientRegion(biome, slopFactor, height);
	if (region == STPTextureSplatRuleWrapper::NoRegion) {
		//no gradient region is being defined, switch to altitude region
		region = splatWrapper.altitudeRegion(biome, height);
		//we don't need to check for null altitude region, if there is none, there is none...
	}
	//write whatever region to the splatmap
	//out-of-boundary write will be caught by CUDA (safely) and will crash the program with error
	surf2Dwrite(region, splatmap_surf, SamplingPosition.x, SamplingPosition.y, cudaBoundaryModeTrap);
}