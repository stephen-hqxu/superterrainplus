#pragma once
#ifndef _STP_SIMPLEX_NOISE_CUH_
#define _STP_SIMPLEX_NOISE_CUH_

//Permutation
#include "../STPPermutation.hpp"

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief Simplex noise is a method for constructing an n-dimensional noise function comparable to
	 * Perlin noise ("classic" noise) but with fewer directional artefacts and, in higher dimensions,
	 * a lower computational overhead. Ken Perlin designed the algorithm in 2001 to address the limitations of his classic noise function,
	 * especially in higher dimensions.
	 * The advantages of simplex noise over Perlin noise:
	 * - Simplex noise has a lower computational complexity and requires fewer multiplications.
	 * - Simplex noise scales to higher dimensions (4D, 5D) with much less computational cost: the complexity is O (n^2)
	 *   for n dimensions instead of the O (n*2^n) of classic noise.
	 * - Simplex noise has no noticeable directional artefacts (is visually isotropic), though noise generated for different dimensions are visually distinct
	 *   (e.g. 2D noise has a different look than slices of 3D noise, and it looks increasingly worse for higher dimensions[citation needed]).
	 * - Simplex noise has a well-defined and continuous gradient (almost) everywhere that can be computed quite cheaply.
	 * - Simplex noise is easy to implement in hardware.
	*/
	namespace STPSimplexNoise {

		/**
		 * @brief STPFractalSimplexInformation specifies parameters for computing fractal simplex noise.
		*/
		struct STPFractalSimplexInformation {
		public:

			//Persistence controls the amplitude multiplier of each octave.
			float Persistence;
			//Lacunarity controls the frequency multiplier of each octave.
			float Lacunarity;
			//Octave denotes the number phase to be performed in a fractal operation.
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
			//The initial frequency of the fractal.
			//In each octave this value will be multiplied by lacunarity.
			float Frequency = 1.0f;
		};

		/**
		 * @brief Generate 2D simplex noise.
		 * @param permutation Provide the permutation table for simplex noise.
		 * @param x X input.
		 * @param y Y input.
		 * @return The simplex noise output.
		*/
		__device__ float simplex2D(const STPPermutation&, float, float);

		/**
		 * @brief Generate fractal 2D simplex noise.
		 * @param permutation Provide the permutation table for simplex noise.
		 * @param x The X input.
		 * @param y The Y input.
		 * @param desc The pointer to the launch information of the simplex fractal.
		 * @return The normalised fractal noise result, in range [0.0, 1.0].
		*/
		__device__ float simplex2DFractal(const STPPermutation&, float, float, const STPFractalSimplexInformation&);

	}
}
#endif//_STP_SIMPLEX_NOISE_CUH_