#pragma once
#ifndef _STP_PERMUTATION_GENERATOR_CUH_
#define _STP_PERMUTATION_GENERATOR_CUH_

//CUDA Runtime
#include <cuda_runtime.h>
//System
#include <random>

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
		 * @brief Generate a random permutaion and gradient table for simplex noise generator
		*/
		class STPPermutationGenerator {
		private:

			//TODO Choose your prefered rng here!!!
			typedef std::mt19937_64 STPPermutationRNG;

			//Generated permutation table, it will be shuffled by rng
			//stored on device
			unsigned char* PERMUTATION = nullptr;

			//Gradient table for simplex noise 2D, the modular of the offset will equal to 1.0
			double* GRADIENT2D = nullptr;

			//The size of the gradient table
			const unsigned int GRADIENT2D_SIZE;

		public:

			/**
			 * @brief Init thhe permutation generator
			 * @param seed Proving the seed for the program
			 * @param distribution Set how many gradient stretch will have, default is 8, each of them will be 45 degree apart
			 * @param offset Set the offset of the angle for the gradient table, in degree
			*/
			__host__ STPPermutationGenerator(unsigned long long, unsigned int = 8u, double = 0.0);

			//Copy permutation generator, deep copy for generated gradient and permutation will be performed.
			__host__ STPPermutationGenerator(const STPPermutationGenerator&) = delete;

			__host__ STPPermutationGenerator(STPPermutationGenerator&&) = delete;

			//Copy the permutation to the destination class, deep copy for generated gradient and permutation will be performed.
			__host__ STPPermutationGenerator& operator=(const STPPermutationGenerator&) = delete;

			__host__ STPPermutationGenerator& operator=(STPPermutationGenerator&&) = delete;

			__host__ ~STPPermutationGenerator();

			/**
			 * @brief Return the randomly generated permutation element from the class generated table.
			 * @param index The index within the table
			 * @return The number, ranged from 0-255 inclusive
			*/
			__device__ int perm(int) const;

			/**
			 * @brief Return the graident table element
			 * @param index The index within the table
			 * @param component The vector component, either 0 or 1 for the 2D table
			 * @return The gradient number
			*/
			__device__ double grad2D(int, int) const;

			/**
			 * @brief Get the number of element in the gradient 2D table
			 * @return The number of element
			*/
			__device__ unsigned int grad2D_size() const;

		};
	}
}
#endif//_STP_PERMUTATION_GENERATOR_CUH_