#ifndef _STP_SIMPLEX_NOISE_RTC_CUH_
#define _STP_SIMPLEX_NOISE_RTC_CUH_

#ifndef __CUDACC__
#error __FILE__ can only be compiled on nvcc and nvrtc exclusively
#endif

#ifndef __CUDACC_RTC__
//allow user to init STPSimplexNoiseRTC on host if they wish to
#define STP_SIMPLEXRTC_HOSTAPI __host__
#else
#define STP_SIMPLEXRTC_HOSTAPI
#endif//__CUDACC_RTC__

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

		class STPSimplexNoise;

		/**
		 * @brief STPSimplexNoiseRTC is a NVRTC adapter of STPSimplexNoise.
		 * @see STPSimplexNoise
		*/
		class STPSimplexNoiseRTC {
		private:

			//Define the device pointer to simplex noise implementation
			const STPSimplexNoise* Impl;

		public:

			/**
			 * @brief Init STPSimplexNoiseRTC with concrete implementation
			 * @impl provide a device pointer to an initialised simplex noise generator
			*/
			STP_SIMPLEXRTC_HOSTAPI __device__ STPSimplexNoiseRTC(const STPSimplexNoise*);

			STP_SIMPLEXRTC_HOSTAPI __device__ ~STPSimplexNoiseRTC();

			/**
			* @brief Generate 2D simplex noise using the provided implementation pointer
			* @param x X input
			* @param y Y input
			* @return The 2D simplex noise result
			*/
			__device__ float simplex2D(float, float) const;

		};

	}
}
//we don't want to expose simplex noise declaration to NVRTC which doesn't accept host functions
#ifndef __CUDACC_RTC__
#include "STPSimplexNoise.cuh"
#endif//__CUDACC_RTC__
#endif//_STP_SIMPLEX_NOISE_RTC_CUH_