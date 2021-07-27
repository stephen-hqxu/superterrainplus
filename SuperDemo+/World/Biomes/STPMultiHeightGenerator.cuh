#pragma once
#ifndef _STP_MULTIHEIGHT_GENERATOR_CUH_
#define _STP_MULTIHEIGHT_GENERATOR_CUH_

//Simplex Noise
#include <SuperAlgorithm+/STPSimplexNoise.cuh>
#include "STPBiomeSettings.hpp"
//CUDA
#include "cuda_runtime.h"

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPMultiHeightGenerator is a wrapper class to the device multi-heightmap generator
	*/
	class STPMultiHeightGenerator {
	private:

		STPMultiHeightGenerator() = default;

		~STPMultiHeightGenerator() = default;

	public:

		/**
		 * @brief Init the multi-height generator
		 * @param biome_settings The biome settings
		 * @param simplex_generator The simplex noise generator, the original ptr must be retained by caller
		 * @param dimension The dimension of the map generated
		*/
		__host__ static void initGenerator(const STPBiomeSettings*, const SuperTerrainPlus::STPCompute::STPSimplexNoise*, uint2);

		/**
		 * @brief Start the generation of multi-biome heightmap
		 * @param heightmap The heightmap ouput
		 * @param dimension The heightmap size
		 * @param offset The heightmap offset
		 * @param stream The cuda stream
		*/
		__host__ static void generateHeightmap(float*, uint2, float2, cudaStream_t);
	};

	

}
#endif//_STP_MULTIHEIGHT_GENERATOR_CUH_