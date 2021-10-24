#pragma once
#ifndef __CUDACC__
#error __FILE__ can only be compiled by nvcc and nvrtc exclusively
#endif

#ifndef _STP_KERNEL_MATH_CUH_
#define _STP_KERNEL_MATH_CUH_

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif//__CUDACC_RTC__

namespace SuperTerrainPlus::STPCompute {

	/**
	 * @brief STPKernelMath is a library contains common math functions. It only serves to kernel
	*/
	namespace STPKernelMath {

		/**
		 * @brief Perform linear interpolation of two data points.
		 * It simply connects two points with a straight line and use the fact to determine the location on the line.
		 * @param p1 The first point of the line
		 * @param p2 The second point of the line
		 * @param factor The normalised distance to p2, the greater, the more p2 value is get.
		 * It must be [0,1]
		 * @return The lerp value of two points with factor
		*/
		__device__ float lerp(float, float, float);

		/**
		 * @brief Perform inverse linear interpolation for each value to scale it within [0,1]
		 * @param minVal The mininmum value
		 * @param maxVal The maximum value
		 * @param value The input value
		 * @return The interpolated value
		*/
		__device__ float Invlerp(float, float, float);

		/**
		 * @brief Perform cosine interpolation for two data points.
		 * It connects two points with a cosine function, and sample the output based on the factor.
		 * @param p1 The first input point
		 * @param p2 The second input point
		 * @param factor The normalised distance to p2.
		 * @return The cosrp value of two points with factor
		*/
		__device__ float cosrp(float, float, float);

		/**
		 * @brief Clamp the value between two ranges.
		 * @param value The value to be clamped
		 * @param min The minumum edge
		 * @param max The maximum edge
		 * @return The clampped result.
		 * If value is greater than upper, upper is returned.
		 * If value is less than lower, lower is returned.
		*/
		__device__ float clamp(float, float, float);

	};
}
#endif//_STP_KERNEL_MATH_CUH_