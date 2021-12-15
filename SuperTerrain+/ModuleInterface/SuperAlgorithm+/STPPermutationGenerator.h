#pragma once
#ifndef _STP_PERMUTATION_GENERATOR_H_
#define _STP_PERMUTATION_GENERATOR_H_

#include <SuperAlgorithm+/STPAlgorithmDefine.h>
#include "STPSimplexNoiseSetting.h"
//System
#include <random>
//Permutation Table
#include "STPPermutation.hpp"
//Device Memory Management
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceMemory.h>

namespace SuperTerrainPlus::STPCompute {

	/**
	 * @brief Generate a random permutaion and gradient table for simplex noise generator
	*/
	class STP_ALGORITHM_HOST_API STPPermutationGenerator : private STPPermutation {
	private:

		//TODO Choose your prefered rng here!!!
		typedef std::mt19937_64 STPPermutationRNG;

		//Manage the memory smartly and only pass the pointer to the STPPermutation
		STPSmartDeviceMemory::STPDeviceMemory<unsigned char[]> ManagedPermutation;
		STPSmartDeviceMemory::STPDeviceMemory<float[]> ManagedGradient2D;

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

		~STPPermutationGenerator() = default;

		/**
		 * @brief Get the generated permutation table.
		 * The table returned is bounded to the current generator, if generator is destroyed result will become undefined.
		 * @return Pointer to the permutation table.
		*/
		const STPPermutation& operator*() const;

	};
}
#endif//_STP_PERMUTATION_GENERATOR_H_