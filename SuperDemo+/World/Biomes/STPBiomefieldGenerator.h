#pragma once
#ifndef _STP_BIOMEFIELD_GENERATOR_H_
#define _STP_BIOMEFIELD_GENERATOR_H_

//Multi-biome Heightfield Generator
#include <SuperTerrain+/World/Chunk/STPDiversityGenerator.hpp>
#include "STPCommonCompiler.h"
#include "STPBiome.hpp"
//Biome Interpolation
#include <SuperAlgorithm+Host/STPSingleHistogramFilter.h>
//Pool
#include <SuperTerrain+/Utility/Memory/STPObjectPool.h>

#include <SuperTerrain+/Utility/Memory/STPSmartDeviceObject.h>

//GLM
#include <glm/vec2.hpp>

namespace STPDemo {

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
		mutable SuperTerrainPlus::STPAlgorithm::STPSingleHistogramFilter GenerateBiomeHistogram;

		const STPCommonCompiler& KernelProgram;
		//The entry global function to generate the heightmap
		CUfunction GeneratorEntry;
		SuperTerrainPlus::STPSmartDeviceObject::STPMemPool HistogramCacheDevice;

		const unsigned int InterpolationRadius;

		/**
		 * @brief STPHistogramBufferCreator creates a new single histogram buffer.
		*/
		struct STPHistogramBufferCreator {
		private:

			//Specifies type of buffer to be created.
			const SuperTerrainPlus::STPAlgorithm::STPSingleHistogramFilter::STPFilterBuffer::STPExecutionType BufferExecution;

		public:

			auto operator()() const;

			/**
			 * @brief Initialise the histogram buffer creator.
			 * @param mapDim The size of the map whose histogram to be generated.
			*/
			STPHistogramBufferCreator(const glm::uvec2&) noexcept;

			~STPHistogramBufferCreator() = default;

		};
		//A queue of histogram buffer
		SuperTerrainPlus::STPObjectPool<SuperTerrainPlus::STPAlgorithm::STPSingleHistogramFilter::STPFilterBuffer,
			STPHistogramBufferCreator> BufferPool;

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

		~STPBiomefieldGenerator() override = default;

		void operator()(const SuperTerrainPlus::STPNearestNeighbourHeightFloatWTextureBuffer&,
			const SuperTerrainPlus::STPNearestNeighbourSampleRTextureBuffer&, glm::vec2) override;

	};

}
#endif//_STP_BIOMEFIELD_GENERATOR_H_