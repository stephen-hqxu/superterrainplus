#pragma once
#ifndef STP_REALISM_IMPLEMENTATION
#error __FILE__ is an internal utility for generating noise texture and should not be used externally
#endif//STP_REALISM_IMPLEMENTATION

#ifndef _STP_RANDOM_TEXTURE_GENERATOR_CUH_
#define _STP_RANDOM_TEXTURE_GENERATOR_CUH_

#include "../Object/STPTexture.h"

#include <SuperTerrain+/World/STPWorldMapPixelFormat.hpp>

//CUDA
#include <cuda_runtime.h>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPRandomTextureGenerator is a simply utility for generating random noise texture on GPU, it supports generating different texture format.
	*/
	namespace STPRandomTextureGenerator {

		/**
		 * @brief Generate a random texture.
		 * @tparam T The type of the texture to be generated.
		 * @param output The output GL texture object where the generated data will be stored.
		 * The generation size will be inferred from the underlying texture directly.
		 * @param seed The seed for generating the noise texture.
		 * @param min The minimum value the noise should have.
		 * @param max The maximum value the noise should have.
		 * @param stream An optional CUDA stream to be used.
		*/
		template<typename T>
		__host__ void generate(const STPTexture&, STPSeed_t, T, T, cudaStream_t);

	}

}
#endif//_STP_RANDOM_TEXTURE_GENERATOR_CUH_