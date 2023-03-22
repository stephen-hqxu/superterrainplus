#pragma once
#ifndef _STP_PERMUTATION_GENERATOR_H_
#define _STP_PERMUTATION_GENERATOR_H_

#include <SuperAlgorithm+Host/STPAlgorithmDefine.h>
#include "STPSimplexNoiseSetting.h"
//Device Memory Management
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceMemory.h>
//Permutation Table
#include "STPPermutation.hpp"

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief Generate a random permutation and gradient table for simplex noise function.
	*/
	namespace STPPermutationGenerator {

		/**
		 * @brief Contain the generated permutation table,
		 * and a shallow copy to the memory to allow device access.
		*/
		class STPPermutationTable {
		private:

			friend STP_ALGORITHM_HOST_API STPPermutationTable generate(const STPEnvironment::STPSimplexNoiseSetting&);

			//The original memory region where the permutations are stored.
			//This memory should be retained as long as the permutation table is used.
			struct {
			public:

				STPSmartDeviceMemory::STPDeviceMemory<unsigned char[]> Permutation;
				STPSmartDeviceMemory::STPDeviceMemory<float[]> Gradient2D;

			} DeviceMemory;

		public:

			//The shallow copy of the permutation device memory, can be passed directly to the device.
			STPPermutation Permutation;

		};

		/**
		 * @brief Generate a new permutation table.
		 * @param simplex_setting Arguments to specify the generator behaviour.
		 * @return The result of generation, should be retained by the user.
		*/
		STP_ALGORITHM_HOST_API STPPermutationTable generate(const STPEnvironment::STPSimplexNoiseSetting&);

	}

}
#endif//_STP_PERMUTATION_GENERATOR_H_