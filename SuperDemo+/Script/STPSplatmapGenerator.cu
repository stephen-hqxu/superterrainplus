//-------------------- Constants ----------------------------
constexpr static float 
	//Control the gradient change, greater decreases gradient in general.
	GradientBias = 5.5f, 
	//Control the distance of pixels in the filter kernel.
	KernelRadius = 2.5f, 
	//Control the scale of noise, greater denominator gives smoother noise.
	NoiseScale = 1.0f / 82.5f,
	//Control how much the noise can affect the height value
	NoiseContribution = 0.05f,
	//A.k.a. altitude, but it does not have to be the same as altitude of the terrain.
	//Control sensitivity of gradient responds to altitude change.
	HeightFactor = 875.5f;

//--------------------- Program Start ------------------------
#include "./Script/STPCommonGenerator.cuh"

//SuperAlgorithm+ Device library
#include <STPTextureSplatRuleWrapper.cuh>

using namespace SuperTerrainPlus::STPCompute;
using SuperTerrainPlus::STPDiversity::Sample;

namespace STPTI = SuperTerrainPlus::STPDiversity::STPTextureInformation;

__constant__ STPTI::STPSplatRuleDatabase SplatDatabase[1];

//A simple 2x2 Sobel kernel
constexpr static unsigned int GradientSize = 4u;
constexpr static int2 GradientKernel[GradientSize] = {
	int2{ 0, -1 },//top, 0
	int2{ -1, 0 },//left, 1
	int2{ 1, 0 },//right, 2
	int2{ 0, 1 }//bottom, 3
};

//--------------------- Definition --------------------------

using namespace STPCommonGenerator;

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

	const STPSimplexNoise Simplex(*Permutation);

	//coordinates are normalised
	const uint2 SamplingPosition = make_uint2(
		x + Dimension->x * local_info.LocalChunkCoordinateX,
		y + Dimension->y * local_info.LocalChunkCoordinateY
	);
	const float2 unitUV = make_float2(
		1.0f / (1.0f * RenderedDimension->x),
		1.0f / (1.0f * RenderedDimension->y)
	);
	const float2 UV = make_float2(
		1.0f * SamplingPosition.x * unitUV.x,
		1.0f * SamplingPosition.y * unitUV.y
	);

	float cell[GradientSize];
	//calculate heightmap gradient
	for (unsigned int i = 0u; i < GradientSize; i++) {
		const int2& currentKernel = GradientKernel[i];
		const float2 offsetUV = make_float2(
			unitUV.x * currentKernel.x * KernelRadius,
			unitUV.y * currentKernel.y * KernelRadius
		);
		const float2 SamplingUV = make_float2(
			UV.x + offsetUV.x,
			UV.y + offsetUV.y
		);
		//sample this heightmap value
		cell[i] = tex2D<float>(heightmap_tex, SamplingUV.x, SamplingUV.y) * HeightFactor;
	}

	//calculate gradient using a very simple 2x2 filter, ranged [-1,1]
	const float gradient[3] = {
		cell[0] - cell[3],
		GradientBias,
		cell[1] - cell[2]
	};
	const float slopFactor = 1.0f - (gradient[1] * rnormf(3, gradient));

	//add some simplex noise to the slopFactor and height value, reminder: range is [-1,1]
	const float noise = Simplex.simplex2D(
		(x + local_info.ChunkMapOffsetX) * NoiseScale,
		(y + local_info.ChunkMapOffsetY) * NoiseScale
	) * NoiseContribution;
	//get information about the current position
	const Sample biome = tex2D<Sample>(biomemap_tex, UV.x, UV.y);
	const float height = STPKernelMath::clamp(tex2D<float>(heightmap_tex, UV.x, UV.y) + noise, 0.0f, 1.0f);

	const STPTextureSplatRuleWrapper splatWrapper(SplatDatabase[0]);
	//get regions, we define gradient region outweights altitude region if they overlap
	unsigned int region = splatWrapper.gradientRegion(biome, slopFactor, height);
	if (region == STPTextureSplatRuleWrapper::NoRegion) {
		//no gradient region is being defined, switch to altitude region
		region = splatWrapper.altitudeRegion(biome, height);
		//we don't need to check for null altitude region, if there is none, there is none...
	}
	//write whatever region to the splatmap
	//out-of-boundary write will be caught by CUDA (safely) and will crash the program with error
	surf2Dwrite(static_cast<unsigned char>(region), splatmap_surf, SamplingPosition.x, SamplingPosition.y, cudaBoundaryModeTrap);
}