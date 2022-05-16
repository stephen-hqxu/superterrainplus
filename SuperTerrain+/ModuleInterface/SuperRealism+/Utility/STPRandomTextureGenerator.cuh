#pragma once
#ifndef STP_REALISM_IMPLEMENTATION
#error __FILE__ is an internal utility for generating noise texture and should not be used externally
#endif//STP_REALISM_IMPLEMENTATION

#ifndef _STP_RANDOM_TEXTURE_GENERATOR_CUH_
#define _STP_RANDOM_TEXTURE_GENERATOR_CUH_

//CUDA
#include <cuda_runtime.h>

//GLM
#include <glm/vec3.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPRandomTextureGenerator is a simply utility for generating random noise texture on GPU, it supports generating different texture format.
	*/
	namespace STPRandomTextureGenerator {

		/**
		 * @brief Generate a random texture.
		 * @tparam T The type of the texture to be generated.
		 * @param output The output CUDA array where the generated data will be stored.
		 * @param dimension The dimension of the texture, unused components must be 1.
		 * For example if only a 2D texture is required, z component must be 1.
		 * Use of zero component will return exception.
		 * @param seed The seed for generating the noise texture.
		 * @param min The minimum value the noise should have.
		 * @param max The maximum value the noise should have.
		*/
		template<typename T>
		__host__ void generate(cudaArray_t, glm::uvec3, unsigned long long, T, T);

	}

}
#endif//_STP_RANDOM_TEXTURE_GENERATOR_CUH_