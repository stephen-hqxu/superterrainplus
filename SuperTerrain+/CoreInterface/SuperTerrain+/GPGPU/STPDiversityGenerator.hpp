#pragma once
#ifndef _STP_DIVERSITY_GENERATOR_HPP_
#define _STP_DIVERSITY_GENERATOR_HPP_

//CUDA Runtime
#include <cuda_runtime.h>
//Biome Defines
#include "../World/Diversity/STPBiomeDefine.h"
//Sample Map Free-Slip
#include "../GPGPU/FreeSlip/STPFreeSlipGenerator.cuh"
//GLM
#include <glm/vec2.hpp>

namespace SuperTerrainPlus::STPCompute {

	/**
	 * @brief STPDiversityGenerator is a base class which provides a programmable interface for customised multi-biome heightmap generation
	*/
	class STPDiversityGenerator {
	protected:

		/**
		 * @brief Init diversity generator
		*/
		STPDiversityGenerator() = default;

	public:

		virtual ~STPDiversityGenerator() = default;

		/**
		 * @brief Generate a biome-specific heightmaps.
		 * Note that this function does not provide any information about the texture size since user is responsible for initialising
		 * the generator like STPHeightfieldGenerator with said parameters, and should keep track on those their own.
		 * @param heightmap_buffer The result of generated heightmap that will be stored, with auto-managed texture memory
		 * @param biomemap_adaptor The free-slip manager adaptor loaded with biomemap,
		 * which is an array of biomeID, the meaning of biomeID is however implementation-specific.
		 * The biomemap data is read-only, writing to the biome map will not affect the original data
		 * Biomemap uses free-slip neighbour logic, the exact number of free-slip chunk is defined by the parameters used in heightfield generator.
		 * @param offset The offset of maps in world coordinate
		 * @param stream The stream currently being used
		*/
		virtual void operator()(STPFreeSlipFloatTextureBuffer&, const STPFreeSlipGenerator::STPFreeSlipSampleManagerAdaptor&, glm::vec2, cudaStream_t) const = 0;

	};

}
#endif//_STP_DIVERSITY_GENERATOR_HPP_