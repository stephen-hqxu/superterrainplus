#pragma once
#ifndef _STP_SIMPLEX_NOISE_CUH_
#define _STP_SIMPLEX_NOISE_CUH_

//Helpers and Tools
#include "STPPermutationsGenerator.cuh"
//Settings
#include "../Settings/STPSimplexNoiseSettings.hpp"

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
		 * @brief Simplex noise is a method for constructing an n-dimensional noise function comparable to
		 * Perlin noise ("classic" noise) but with fewer directional artifacts and, in higher dimensions, 
		 * a lower computational overhead. Ken Perlin designed the algorithm in 2001[1] to address the limitations of his classic noise function, 
		 * especially in higher dimensions. 
		 * The advantages of simplex noise over Perlin noise:
		 * - Simplex noise has a lower computational complexity and requires fewer multiplications.
		 * - Simplex noise scales to higher dimensions (4D, 5D) with much less computational cost: the complexity is O ( n 2 ) {\displaystyle O(n^{2})} O(n^{2}) 
		 *   for n {\displaystyle n} n dimensions instead of the O ( n 2 n ) {\displaystyle O(n\,2^{n})} {\displaystyle O(n\,2^{n})} of classic noise.[2]
		 * - Simplex noise has no noticeable directional artifacts (is visually isotropic), though noise generated for different dimensions are visually distinct 
		 *   (e.g. 2D noise has a different look than slices of 3D noise, and it looks increasingly worse for higher dimensions[citation needed]).
		 * - Simplex noise has a well-defined and continuous gradient (almost) everywhere that can be computed quite cheaply.
		 * - Simplex noise is easy to implement in hardware.
		*/
		class STPSimplexNoise : private STPPermutationsGenerator{
		public:

			/**
			 * @brief Init the simplex noise generator.
			 * @param noise_settings Provide the settings for simplex noise
			*/
			__host__ STPSimplexNoise(const STPSettings::STPSimplexNoiseSettings* const);

			__host__ STPSimplexNoise(const STPSimplexNoise&) = delete;

			__host__ STPSimplexNoise(STPSimplexNoise&&) = delete;

			__host__ STPSimplexNoise& operator=(const STPSimplexNoise&) = delete;

			__host__ STPSimplexNoise& operator=(STPSimplexNoise&&) = delete;

			__host__ ~STPSimplexNoise();

			/**
			 * @brief Generate 2D simplex noise
			 * @param x X intput
			 * @param y Y input
			 * @return The simplex noise output
			*/
			__device__ float simplex2D(float, float) const;

		};
	}
}
#endif//_STP_SIMPLEX_NOISE_CUH_