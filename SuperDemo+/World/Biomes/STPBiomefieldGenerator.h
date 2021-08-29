#pragma once
#ifndef _STP_BIOMEFIELD_GENERATOR_H_
#define _STP_BIOMEFIELD_GENERATOR_H_

//System
#include <queue>
#include <mutex>
//Multi-biome Heightfield Generator
#include <GPGPU/STPDiversityGeneratorRTC.h>
#include <SuperAlgorithm+/STPPermutationGenerator.h>
#include "STPBiomeSettings.hpp"
//Biome Interpolation
#include <SuperAlgorithm+/STPSingleHistogramFilter.h>
#include <glm/vec2.hpp>

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPBiomefieldGenerator is a sample implementation of multi-biome heightfield generator.
	 * It generates different heightfield based on biome settings.
	 * Heightfield generator uses NVRTC
	*/
	class STPBiomefieldGenerator : public SuperTerrainPlus::STPCompute::STPDiversityGeneratorRTC {
	private:

		//all parameters for the noise generator, stored on host, passing value to device
		SuperTerrainPlus::STPEnvironment::STPSimplexNoiseSetting Noise_Setting;
		SuperTerrainPlus::STPCompute::STPPermutationGenerator Simplex_Permutation;
		//The size of the generated heightmap
		const uint2 MapSize;
		//Generate a histogram to retrieve weights for biomes in a range
		mutable std::mutex HistogramFilterLock;
		mutable SuperTerrainPlus::STPCompute::STPSingleHistogramFilter biome_histogram;

		CUcontext cudaCtx;
		//The entry global function to generate the heightmap
		CUfunction GeneratorEntry;
		CUmemoryPool HistogramCacheDevice;

		const unsigned int InterpolationRadius;

		//A queue of histogram buffer
		typedef std::queue<STPSingleHistogramFilter::STPHistogramBuffer_t> STPHistogramBufferPool;
		mutable STPHistogramBufferPool BufferPool;
		mutable std::mutex BufferPoolLock;

		/**
		 * @brief Init the multi-height generator
		*/
		void initGenerator();

		//Essential data to return the histogram buffer back to pool after the computation has finished
		struct STPBufferReleaseData;

		//A CUDA stream call back to return histogram buffer back to the pool
		static void returnBuffer(void*);

	public:

		/**
		 * @brief Init the demo biomefield generator
		 * @param simplex_setting The settings for the simplex noise generator
		 * @param dimension The size of the generated heightmap
		 * @param interpolation_radius The radius for biome edge interpolation
		*/
		STPBiomefieldGenerator(SuperTerrainPlus::STPEnvironment::STPSimplexNoiseSetting&, glm::uvec2, unsigned int);

		STPBiomefieldGenerator(const STPBiomefieldGenerator&) = delete;

		STPBiomefieldGenerator(STPBiomefieldGenerator&&) = delete;

		STPBiomefieldGenerator& operator=(const STPBiomefieldGenerator&) = delete;

		STPBiomefieldGenerator& operator=(STPBiomefieldGenerator&&) = delete;

		~STPBiomefieldGenerator();

		void operator()(SuperTerrainPlus::STPCompute::STPFreeSlipFloatTextureBuffer&, const SuperTerrainPlus::STPCompute::STPFreeSlipGenerator::STPFreeSlipSampleManagerAdaptor&, float2, cudaStream_t) const override;

	};

}
#endif//_STP_BIOMEFIELD_GENERATOR_H_