#ifndef _STP_PERMUTATION_HPP_
#define _STP_PERMUTATION_HPP_

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
		 * @brief STPPermutation is a device lookup table for simplex noise, generated and managed by STPPermutationGenerator
		*/
		struct STPPermutation {
		public:

			//Generated permutation table, it will be shuffled by rng
			//stored on device
			unsigned char* Permutation = nullptr;

			//Gradient table for simplex noise 2D, the modular of the offset will equal to 1.0
			float* Gradient2D = nullptr;

			//The size of the gradient table
			unsigned int Gradient2DSize;

		};

	}
}
#endif//_STP_PERMUTATION_HPP_