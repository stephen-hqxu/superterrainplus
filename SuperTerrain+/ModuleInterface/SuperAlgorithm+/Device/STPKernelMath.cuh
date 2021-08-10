#pragma once
#ifndef __CUDACC__
#error __FILE__ can only be compiled by nvcc and nvrtc exclusively
#endif

#ifndef _STP_KERNEL_MATH_CUH_
#define _STP_KERNEL_MATH_CUH_

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
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

		/**
		 * @brief STPKernelMath is a library contains common math functions. It only serves to kernel
		*/
		class STPKernelMath {
		private:

			/**
			 * @brief It's a static-only class so don't initialise
			*/
			__device__ STPKernelMath() = delete;

			__device__ ~STPKernelMath() = delete;

		public:

			/**
			 * @brief Perform linear interpolation of two data points.
			 * It simply connects two points with a straight line and use the fact to determine the location on the line.
			 * @param p1 The first point of the line
			 * @param p2 The second point of the line
			 * @param factor The normalised distance to p2, the greater, the more p2 value is get.
			 * It must be [0,1]
			 * @return The lerp value of two points with factor
			*/
			__device__ static float lerp(float, float, float);

			/**
			 * @brief Perform inverse linear interpolation for each value to scale it within [0,1]
			 * @param minVal The mininmum value
			 * @param maxVal The maximum value
			 * @param value The input value
			 * @return The interpolated value
			*/
			__device__ static float Invlerp(float, float, float);

			/**
			 * @brief Perform cosine interpolation for two data points.
			 * It connects two points with a cosine function, and sample the output based on the factor.
			 * @param p1 The first input point
			 * @param p2 The second input point
			 * @param factor The normalised distance to p2.
			 * @return The cosrp value of two points with factor
			*/
			__device__ static float cosrp(float, float, float);

			/**
			 * @brief Clamp the value between two ranges.
			 * @param value The value to be clamped
			 * @param min The minumum edge
			 * @param max The maximum edge
			 * @return The clampped result.
			 * If value is greater than upper, upper is returned.
			 * If value is less than lower, lower is returned.
			*/
			__device__ static float clamp(float, float, float);

		};

	}
}
#endif//_STP_KERNEL_MATH_CUH_