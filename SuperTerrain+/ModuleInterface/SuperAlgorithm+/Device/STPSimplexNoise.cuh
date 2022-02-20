#pragma once
#ifndef __CUDACC__
#error __FILE__ can only be compiled by nvcc and nvrtc exclusively
#endif

#ifndef _STP_SIMPLEX_NOISE_CUH_
#define _STP_SIMPLEX_NOISE_CUH_

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif//__CUDACC_RTC__

//Permutation
#include "../STPPermutation.hpp"

namespace SuperTerrainPlus::STPCompute {

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
	class STPSimplexNoise {
	private:

		const STPPermutation& Permutation;

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
		__device__ float grad2D(int, int) const;

	public:

		/**
		 * @brief STPFractalSimplexInformation specifies parameters for computing fractal simplex noise.
		*/
		struct STPFractalSimplexInformation {
		public:

			//Persistence controls the amplitude multiplier of each octave.
			float Persistence;
			//Lacunarity controls the frequency multiplier of each octave.
			float Lacunarity;
			//Octave denotes the number phase to be performned in a fractal operation.
			unsigned int Octave;
			//The half dimension of the generated texture using simplex noise fractal.
			//By using half dimension, noise is scaled at the centre of the image instead of the edge.
			float2 HalfDimension;
			//Specify the offset of the noise.
			float2 Offset;
			//Specify the scale of the noise.
			float Scale;

			/* The following variables will be changed at the end of the execution, as outputs */
			
			//The initial amplitude of the fractal.
			//In each octave this value will be multiplied by persistence.
			float Amplitude = 1.0f;
			//The initial freqency of the fractal.
			//In each octave this value will be multiplied by lacunarity.
			float Frequency = 1.0f;
		};

		/**
		 * @brief Init the simplex noise generator.
		 * @param permutation Provide the permutation table for simplex noise.
		*/
		__device__  STPSimplexNoise(const STPPermutation&);

		__device__ STPSimplexNoise(const STPSimplexNoise&) = delete;

		__device__ STPSimplexNoise(STPSimplexNoise&&) = delete;

		__device__ STPSimplexNoise& operator=(const STPSimplexNoise&) = delete;

		__device__ STPSimplexNoise& operator=(STPSimplexNoise&&) = delete;

		__device__ ~STPSimplexNoise();

		/**
		 * @brief Generate 2D simplex noise
		 * @param x X intput
		 * @param y Y input
		 * @return The simplex noise output
		*/
		__device__ float simplex2D(float, float) const;

		/**
		 * @brief Generate fractal 2D simplex noise.
		 * @param x The X input.
		 * @param y The Y input.
		 * @param desc The pointer to the launch information of the simplex fractal
		 * @return The resultant unnormalised fractal noise as the first component, with min and max value of the noise for the rests.
		 * The min and max value can be used for clampping the noise value.
		*/
		__device__ float3 simplex2DFractal(float, float, STPFractalSimplexInformation&) const;

	};
}
#endif//_STP_SIMPLEX_NOISE_CUH_