#pragma once
#ifndef _STP_PERMUTATIONS_GENERATOR_CUH_
#define _STP_PERMUTATIONS_GENERATOR_CUH_

//CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//System
#include <random>
#include <math.h>

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
		class STPPermutationsGenerator {
		private:

			//TODO Choose your prefered rng here!!!
			typedef std::mt19937_64 STPPermutationRNG;

			static constexpr double PI = 3.14159265358979323846;
			//Initial table, will be shuffled later
			static constexpr unsigned char INIT_TABLE[256] = {
				0,1,2,3,4,5,6,7,8,9,10,
				11,12,13,14,15,16,17,18,19,20,
				21,22,23,24,25,26,27,28,29,30,
				31,32,33,34,35,36,37,38,39,40,
				41,42,43,44,45,46,47,48,49,50,
				51,52,53,54,55,56,57,58,59,60,
				61,62,63,64,65,66,67,68,69,70,
				71,72,73,74,75,76,77,78,79,80,
				81,82,83,84,85,86,87,88,89,90,
				91,92,93,94,95,96,97,98,99,100,
				101,102,103,104,105,106,107,108,109,110,
				111,112,113,114,115,116,117,118,119,120,
				121,122,123,124,125,126,127,128,129,130,
				131,132,133,134,135,136,137,138,139,140,
				141,142,143,144,145,146,147,148,149,150,
				151,152,153,154,155,156,157,158,159,160,
				161,162,163,164,165,166,167,168,169,170,
				171,172,173,174,175,176,177,178,179,180,
				181,182,183,184,185,186,187,188,189,190,
				191,192,193,194,195,196,197,198,199,200,
				201,202,203,204,205,206,207,208,209,210,
				211,212,213,214,215,216,217,218,219,220,
				221,222,223,224,225,226,227,228,229,230,
				231,232,233,234,235,236,237,238,239,240,
				241,242,243,244,245,246,247,248,249,250,
				251,252,253,254,255
			};
			//Generated permutation table, it will be shuffled by rng
			//stored on device
			unsigned char* PERMUTATIONS = nullptr;

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
			__host__ STPPermutationsGenerator(unsigned long long, unsigned int = 8u, double = 0.0);

			//Copy permutation generator, deep copy for generated gradient and permutation will be performed.
			__host__ STPPermutationsGenerator(const STPPermutationsGenerator&) = delete;

			__host__ STPPermutationsGenerator(STPPermutationsGenerator&&) = delete;

			//Copy the permutation to the destination class, deep copy for generated gradient and permutation will be performed.
			__host__ STPPermutationsGenerator& operator=(const STPPermutationsGenerator&) = delete;

			__host__ STPPermutationsGenerator& operator=(STPPermutationsGenerator&&) = delete;

			__host__ ~STPPermutationsGenerator();

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
#endif//_STP_PERMUTATIONS_GENERATOR_CUH_