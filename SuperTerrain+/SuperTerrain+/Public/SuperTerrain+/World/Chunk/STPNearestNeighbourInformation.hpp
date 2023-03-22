#pragma once
#ifndef _STP_NEAREST_NEIGHBOUR_INFORMATION_HPP_
#define _STP_NEAREST_NEIGHBOUR_INFORMATION_HPP_

//GLM
#include <glm/vec2.hpp>

namespace SuperTerrainPlus {

	/**
	 * @brief STPNearestNeighbourInformation holds information for nearest neighbour chunk data.
	*/
	struct STPNearestNeighbourInformation {
	public:

		//The dimension of map of a chunk.
		glm::uvec2 MapSize;
		//The number of neighbour chunk around the centre chunk.
		glm::uvec2 ChunkNearestNeighbour;

		//The total dimension of map including the centre chunk and nearest neighbours.
		//This should equal to the product of map size of each chunk and total number of nearest neighbour, as indicated by the parameters above.
		glm::uvec2 TotalMapSize;

	};

}
#endif//_STP_NEAREST_NEIGHBOUR_INFORMATION_HPP_