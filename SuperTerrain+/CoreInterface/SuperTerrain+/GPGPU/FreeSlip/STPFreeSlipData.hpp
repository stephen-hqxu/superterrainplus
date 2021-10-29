#pragma once
#ifndef _STP_FREESLIP_DATA_HPP_
#define _STP_FREESLIP_DATA_HPP_

//CUDA vector
#include <vector_types.h>

//GLM
#include <glm/vec2.hpp>

namespace SuperTerrainPlus::STPCompute {

	/**
	 * @brief STPFreeSlipData holds all data required for free-slip indexing.
	 * Copy to this class will shallow copy the index table pointer, it's recommend using STPFreeSlipGenerator to manage all data
	*/
	struct STPFreeSlipData {
	public:

		/**
		 * @brief Convert global index to local index, making data access outside the central chunk available
		 * As shown the difference between local and global index
		 *		 Local					 Global
		 * 0 1 2 3 | 0 1 2 3	0  1  2  3  | 4  5  6  7
		 * 4 5 6 7 | 4 5 6 7	8  9  10 11 | 12 13 14 15
		 * -----------------	-------------------------
		 * 0 1 2 3 | 0 1 2 3	16 17 18 19 | 20 21 22 23
		 * 4 5 6 7 | 4 5 6 7	24 25 26 27 | 28 29 30 21
		 *
		 * Given that chunk should be arranged in a linear array (just an example)
		 * Chunk 1 | Chunk 2
		 * -----------------
		 * Chunk 3 | Chunk 4
		*/
		unsigned int* GlobalLocalIndex;

		//The dimension of each map
		glm::uvec2 Dimension;
		//The range of free slip in the unit of chunk
		glm::uvec2 FreeSlipChunk;
		//number of element in a global row and column in the free slip range
		glm::uvec2 FreeSlipRange;

	};

}
#endif//_STP_FREESLIP_DATA_HPP_