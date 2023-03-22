#ifndef _STP_PERMUTATION_HPP_
#define _STP_PERMUTATION_HPP_

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPPermutation is a device lookup table for simplex noise, generated and managed by STPPermutationGenerator
	*/
	struct STPPermutation {
	public:

		//Generated permutation table, it will be shuffled by rng
		//stored on device
		unsigned char* Permutation;

		//Gradient table for simplex noise 2D, the modular of the offset will equal to 1.0
		float* Gradient2D;

		//The size of the gradient table
		unsigned int Gradient2DSize;

	};

}
#endif//_STP_PERMUTATION_HPP_