#pragma once
#ifndef _STP_BIOMEFIELD_GENERATOR_H_
#define _STP_BIOMEFIELD_GENERATOR_H_

#include <glm/vec2.hpp>
//Multi-biome Heightfield Generator
#include <GPGPU/STPDiversityGeneratorRTC.h>
#include <SuperAlgorithm+/STPPermutationGenerator.h>
#include "STPBiomeSettings.hpp"

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

		//The entry global function to generate the heightmap
		CUfunction GeneratorEntry;

		/**
		 * @brief Init the multi-height generator
		 * @param biome_settings The biome settings
		*/
		void initGenerator(const STPBiomeSettings*);

	public:

		/**
		 * @brief Init the demo biomefield generator
		 * @param simplex_setting The settings for the simplex noise generator
		 * @param dimension The size of the generated heightmap
		*/
		STPBiomefieldGenerator(SuperTerrainPlus::STPEnvironment::STPSimplexNoiseSetting&, glm::uvec2);

		STPBiomefieldGenerator(const STPBiomefieldGenerator&) = delete;

		STPBiomefieldGenerator(STPBiomefieldGenerator&&) = delete;

		STPBiomefieldGenerator& operator=(const STPBiomefieldGenerator&) = delete;

		STPBiomefieldGenerator& operator=(STPBiomefieldGenerator&&) = delete;

		~STPBiomefieldGenerator() = default;

		void operator()(float*, const Sample*, float2, cudaStream_t) const override;

	};

}
#endif//_STP_BIOMEFIELD_GENERATOR_H_