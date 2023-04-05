//-------------------- Constants ----------------------------
constexpr static float 
	//Control the gradient change, greater decreases gradient in general.
	GradientBias = 5.5f, 
	//Control the distance of pixels in the filter kernel.
	KernelRadius = 2.5f, 
	//Control the scale of noise, greater denominator gives smoother noise.
	NoiseScale = 91.5f,
	//Control how much the noise can affect the height value
	NoiseContribution = 0.15f,
	//A.k.a. altitude, but it does not have to be the same as altitude of the terrain.
	//Control sensitivity of gradient responds to altitude change.
	HeightFactor = 875.5f;

//Simplex noise fractal settings
constexpr static float
	Per = 0.7f,
	Lac = 1.88f;

//--------------------- Program Start ------------------------
#include "./Script/STPCommonGenerator.cuh"

//SuperAlgorithm+ Device library
#include <SuperAlgorithm+Device/STPTextureSplatRuleWrapper.cuh>

#include <SuperTerrain+/Utility/STPDeviceLaunchSetup.cuh>

using namespace SuperTerrainPlus::STPAlgorithm;
using SuperTerrainPlus::STPSample_t, SuperTerrainPlus::STPRegion_t;

namespace STPTI = SuperTerrainPlus::STPDiversity::STPTextureInformation;

__constant__ STPTI::STPSplatRuleDatabase SplatDatabase[1];

//A simple 2x2 kernel
constexpr static unsigned int GradientSize = 4u;
constexpr static int2 GradientKernel[GradientSize] = {
	int2{ 0, -1 },//top, 0
	int2{ -1, 0 },//left, 1
	int2{ 1, 0 },//right, 2
	int2{ 0, 1 }//bottom, 3
};

/**
 * @brief Generate some simplex noise for an input.
 * @param x The x coordinate
 * @param y The y coordinate
 * @param offset The noise offset
 * @return The normalised noise with noise contribution applied.
*/
__device__ static float generateNoise(unsigned int, unsigned int, float2);

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
__global__ void generateTextureSplatmap(const cudaTextureObject_t biomemap_tex, const cudaTextureObject_t heightmap_tex,
	const cudaSurfaceObject_t splatmap_surf, const STPTI::STPSplatGeneratorInformation splat_info) {
	//block is in 2D, so threadIdx.z is always 0 and blockDim.z is always 1
	const auto [x, y, z] = SuperTerrainPlus::STPDeviceLaunchSetup::calcThreadIndex<3u>();
	if (x >= Dimension->x || y >= Dimension->y || z >= splat_info.LocalCount) {
		return;
	}
	//working pixel
	//we need to convert z-coord of thread to chunk local ID
	const STPTI::STPSplatGeneratorInformation::STPLocalChunkInformation& local_info = splat_info.RequestingLocalInfo[z];

	//coordinates are un-normalised
	const uint2 SamplingPosition = make_uint2(
		x + Dimension->x * local_info.LocalChunkCoordinateX,
		y + Dimension->y * local_info.LocalChunkCoordinateY
	);

	float cell[GradientSize];
	//calculate heightmap gradient
	for (unsigned int i = 0u; i < GradientSize; i++) {
		const int2& currentKernel = GradientKernel[i];
		const float2 offsetUV = make_float2(
			currentKernel.x * KernelRadius,
			currentKernel.y * KernelRadius
		);
		const float2 SamplingUV = make_float2(
			SamplingPosition.x + offsetUV.x,
			SamplingPosition.y + offsetUV.y
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
	const float noise = generateNoise(x, y, make_float2(local_info.ChunkMapOffsetX, local_info.ChunkMapOffsetY));
	//get information about the current position
	const STPSample_t biome = tex2D<STPSample_t>(biomemap_tex, SamplingPosition.x, SamplingPosition.y);
	const float height = __saturatef(tex2D<float>(heightmap_tex, SamplingPosition.x, SamplingPosition.y) + noise);

	const STPTextureSplatRuleWrapper splatWrapper(*SplatDatabase);
	const STPTI::STPSplatRegistry* const registry = splatWrapper.findSplatRegistry(biome);
	//get regions, we define gradient region outweighs altitude region if they overlap
	unsigned int region = splatWrapper.gradientRegion(registry, slopFactor, height);
	if (region == STPTextureSplatRuleWrapper::NoRegion) {
		//no gradient region is being defined, switch to altitude region
		region = splatWrapper.altitudeRegion(registry, height);
		//we don't need to check for null altitude region, if there is none, there is none...
	}
	//write whatever region to the splatmap
	//out-of-boundary write will be caught by CUDA (safely) and will crash the program with error
	surf2Dwrite(static_cast<STPRegion_t>(region), splatmap_surf, SamplingPosition.x * sizeof(STPRegion_t),
		SamplingPosition.y, cudaBoundaryModeTrap);
}

__device__ float generateNoise(const unsigned int x, const unsigned int y, const float2 offset) {
	//use simplex noise to generate fractals
	STPSimplexNoise::STPFractalSimplexInformation fractal_info = { };
	fractal_info.Persistence = Per;
	fractal_info.Lacunarity = Lac;
	fractal_info.Octave = 3u;
	fractal_info.Scale = NoiseScale;
	fractal_info.Offset = offset;
	fractal_info.HalfDimension = *HalfDimension;

	return STPSimplexNoise::simplex2DFractal(*Permutation, 1.0f * x, 1.0f * y, fractal_info) * NoiseContribution;
}