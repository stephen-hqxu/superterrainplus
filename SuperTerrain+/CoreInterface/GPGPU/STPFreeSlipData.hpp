#pragma once
#ifndef _STP_FREESLIP_DATA_HPP_
#define _STP_FREESLIP_DATA_HPP_

//CUDA vector
#include <vector_types.h>

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
			uint2 Dimension;
			//The range of free slip in the unit of chunk
			uint2 FreeSlipChunk;
			//number of element in a global row and column in the free slip range
			uint2 FreeSlipRange;

		};

	}
}
#endif//_STP_FREESLIP_DATA_HPP_