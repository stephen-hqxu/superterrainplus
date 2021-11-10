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
	const uint2 SamplingPosition = uint2(
		x + MapDimension->x * local_info.LocalChunkCoordinateX,
		y + MapDimension->y * local_info.LocalChunkCoordinateY
	);
	const float2 UV = float2(
		1.0f * SamplingPosition.x / TotalBufferDimension->x,
		1.0f * SamplingPosition.y / TotalBufferDimension->y
	);
	//get information about the current position
	const Sample biome = tex2D<Sample>(biomemap_tex, UV.x, UV.y);
}