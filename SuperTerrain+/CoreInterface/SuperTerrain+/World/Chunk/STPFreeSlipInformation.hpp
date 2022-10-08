#pragma once
#ifndef _STP_FREESLIP_INFORMATION_HPP_
#define _STP_FREESLIP_INFORMATION_HPP_

//GLM
#include <glm/vec2.hpp>

namespace SuperTerrainPlus {

	/**
	 * @brief STPFreeSlipInformation holds information for free-slip data.
	*/
	struct STPFreeSlipInformation {
	public:

		//The dimension of each map
		glm::uvec2 Dimension;
		//The range of free slip in the unit of chunk
		glm::uvec2 FreeSlipChunk;
		//number of element in a global row and column in the free slip range
		glm::uvec2 FreeSlipRange;

	};

}
#endif//_STP_FREESLIP_INFORMATION_HPP_