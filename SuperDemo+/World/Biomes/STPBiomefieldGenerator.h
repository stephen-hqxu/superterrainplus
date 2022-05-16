#pragma once
#ifndef _STP_BIOMEFIELD_GENERATOR_H_
#define _STP_BIOMEFIELD_GENERATOR_H_

//System
#include <queue>
#include <mutex>
//Multi-biome Heightfield Generator
#include <SuperTerrain+/World/Chunk/STPDiversityGenerator.hpp>
#include "STPCommonCompiler.h"
#include "STPBiome.hpp"
//Biome Interpolation
#include <SuperAlgorithm+/STPSingleHistogramFilter.h>
#include <glm/vec2.hpp>
//GLM
#include <glm/vec2.hpp>

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPBiomefieldGenerator is a sample implementation of multi-biome heightfield generator.
	 * It generates different heightfield based on biome settings.
	 * Heightfield generator uses NVRTC
	*/
	class STPBiomefieldGenerator final : public SuperTerrainPlus::STPDiversityGenerator {
	private:

		//The size of the generated heightmap
		const glm::uvec2 MapSize;
		//Generate a histogram to retrieve weights for biomes in a range
		mutable SuperTerrainPlus::STPAlgorithm::STPSingleHistogramFilter biome_histogram;

		const STPCommonCompiler& KernelProgram;
		//The entry global function to generate the heightmap
		CUfunction GeneratorEntry;
		CUmemoryPool HistogramCacheDevice;

		const unsigned int InterpolationRadius;

		//A queue of histogram buffer
		typedef std::queue<SuperTerrainPlus::STPAlgorithm::STPSingleHistogramFilter::STPHistogramBuffer_t> STPHistogramBufferPool;
		mutable STPHistogramBufferPool BufferPool;
		mutable std::mutex BufferPoolLock;

		/**
		 * @brief Init the multi-height generator
		*/
		void initGenerator();

		//Essential data to return the histogram buffer back to pool after the computation has finished
		struct STPBufferReleaseData;

	public:

		/**
		 * @brief Init the demo biomefield generator.
		 * @param program The compiler that holds the program to the complete biomefield generator kernel.
		 * @param dimension The size of the generated heightmap.
		 * @param interpolation_radius The radius for biome edge interpolation.
		*/
		STPBiomefieldGenerator(const STPCommonCompiler&, glm::uvec2, unsigned int);

		STPBiomefieldGenerator(const STPBiomefieldGenerator&) = delete;

		STPBiomefieldGenerator(STPBiomefieldGenerator&&) = delete;

		STPBiomefieldGenerator& operator=(const STPBiomefieldGenerator&) = delete;

		STPBiomefieldGenerator& operator=(STPBiomefieldGenerator&&) = delete;

		~STPBiomefieldGenerator();

		void operator()(SuperTerrainPlus::STPFreeSlipFloatTextureBuffer&, 
			SuperTerrainPlus::STPFreeSlipSampleTextureBuffer&, 
			const SuperTerrainPlus::STPFreeSlipInformation&, glm::vec2, cudaStream_t) const override;

	};

}
#endif//_STP_BIOMEFIELD_GENERATOR_H_