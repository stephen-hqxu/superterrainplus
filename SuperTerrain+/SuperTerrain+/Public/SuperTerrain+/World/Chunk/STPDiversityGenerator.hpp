#pragma once
#ifndef _STP_DIVERSITY_GENERATOR_HPP_
#define _STP_DIVERSITY_GENERATOR_HPP_

#include "STPNearestNeighbourTextureBuffer.h"

#include <glm/vec2.hpp>

namespace SuperTerrainPlus {

	/**
	 * @brief STPDiversityGenerator is a base class which provides a programmable interface for customised multi-biome heightmap generation
	*/
	class STPDiversityGenerator {
	public:

		/**
		 * @brief Init diversity generator
		*/
		STPDiversityGenerator() = default;

		virtual ~STPDiversityGenerator() = default;

		/**
		 * @brief Generate a multi-biome heightmap.
		 * Both heightmap and biomemap buffer will provide the same information regarding CUDA memory pool, device stream.
		 * The nearest neighbour information, however, are potentially different for both buffer.
		 * The information regarding each map buffer can be found in their corresponding texture buffer.
		 * @param heightmap_buffer The result of generated heightmap that will be stored, with auto-managed texture memory.
		 * The heightmap should be generated for one chunk.
		 * @param biomemap_buffer The nearest-neighbour of the chunk whose heightmap should be generated as specified above, with buffer loaded with biomemap,
		 * which is an array of biomeID, the meaning of biomeID is however implementation-specific.
		 * The biomemap data is read-only, writing to the biome map will not affect the original data.
		 * Biomemap uses nearest neighbour logic.
		 * @param offset The offset of maps in world coordinate.
		*/
		virtual void operator()(const STPNearestNeighbourHeightFloatWTextureBuffer&, const STPNearestNeighbourSampleRTextureBuffer&, glm::vec2) = 0;

	};

}
#endif//_STP_DIVERSITY_GENERATOR_HPP_