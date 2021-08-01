#pragma once
#ifndef _STP_PERMUTATION_GENERATOR_H_
#define _STP_PERMUTATION_GENERATOR_H_

#include <STPAlgorithmDefine.h>
#include "STPSimplexNoiseSetting.h"
//System
#include <random>
//Permutation Table
#include "STPPermutation.hpp"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {
	/**
	 * @brief GPGPU compute suites for Super Terrain + program, powered by CUDA
	*/
	namespace STPCompute {

		/**
		 * @brief Generate a random permutaion and gradient table for simplex noise generator
		*/
		class STPALGORITHMPLUS_HOST_API STPPermutationGenerator {
		private:

			//TODO Choose your prefered rng here!!!
			typedef std::mt19937_64 STPPermutationRNG;

			//Simplex noise generator lookup table, contains permutation and gradient
			STPPermutation Permutation;

		public:

			/**
			 * @brief Init thhe permutation generator
			 * @param simplex_setting Arguments to specify the generator behaviour
			*/
			STPPermutationGenerator(const STPEnvironment::STPSimplexNoiseSetting&);

			//Copy permutation generator, deep copy for generated gradient and permutation will be performed.
			STPPermutationGenerator(const STPPermutationGenerator&) = delete;

			STPPermutationGenerator(STPPermutationGenerator&&) = delete;

			//Copy the permutation to the destination class, deep copy for generated gradient and permutation will be performed.
			STPPermutationGenerator& operator=(const STPPermutationGenerator&) = delete;

			STPPermutationGenerator& operator=(STPPermutationGenerator&&) = delete;

			~STPPermutationGenerator();

			/**
			 * @brief Get the generated permutation table.
			 * The table returned is bounded to the current generator, if generator is destroyed result will become undefined.
			 * @return Pointer to the permutation table.
			*/
			const STPPermutation& operator()() const;

		};
	}
}
#endif//_STP_PERMUTATION_GENERATOR_H_