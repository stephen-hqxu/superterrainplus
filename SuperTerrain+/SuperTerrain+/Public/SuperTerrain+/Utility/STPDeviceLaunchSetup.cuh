#pragma once
#ifndef _STP_DEVICE_LAUNCH_SETUP_CUH_
#define _STP_DEVICE_LAUNCH_SETUP_CUH_

#ifndef __CUDACC_RTC__
#include <SuperTerrain+/STPCoreDefine.h>

#include <glm/vec3.hpp>
#include <cuda_runtime.h>
#include <utility>
#endif

namespace SuperTerrainPlus {

	/**
	 * @brief STPDeviceLaunchSetup contains a number of utilities for setting up device function call.
	*/
	namespace STPDeviceLaunchSetup {

		typedef unsigned char STPDimensionSize;

//Hide host function from NVRTC who cannot compile
#ifndef __CUDACC_RTC__

		//The device kernel launch configuration with grid and block dimension in order.
		typedef std::pair<dim3, dim3> STPLaunchConfiguration;

		namespace STPInternal {

			/**
			 * @brief Normalise thread size of any dimension to 3D.
			 * @tparam Vec The type of thread size vector. Can be a scalar for 1D thread or a vector for 2D and 3D thread.
			 * @param threadSize The thread size to be padded
			 * @return A 3D vector.
			 * Extra dimensions are padded with one.
			 * Dimension axes less than one will be clamped to one.
			*/
			template<class Vec>
			__host__ glm::uvec3 normaliseThreadSize(Vec);

			/**
			 * @brief Similar to the function below, but it does not perform any check.
			 * @param blockDim The dimension of the block.
			 * @param blockSize The block size
			 * @param threadSize The thread size, which is normalised.
			 * @return The launch configuration.
			*/
			__host__ STP_API STPLaunchConfiguration determineLaunchConfiguration(STPDimensionSize, unsigned int, glm::uvec3);

		}

		/**
		 * @brief Determine the launch configuration based on the designated grid and block size in 1D.
		 * @param Block The dimension of block to use.
		 * @tparam Vec A primitive type if it is 1D, or a vector type if it is multi-dimensional.
		 * The vector type must support basic arithmetic operators. Recommended to use GLM vectors.
		 * The dimension of grid to use will then be inferred from the dimension of this vector type.
		 * @param blockSize The block size. Must be a strictly positive integer.
		 * Optimally this should be the value returned from the CUDA occupancy calculator.
		 * @param threadSize The intended thread size to be used for launch.
		 * @return The launch configurations.
		*/
		template<STPDimensionSize Block, class Vec>
		__host__ STPLaunchConfiguration determineLaunchConfiguration(int, const Vec&);

		/**
		 * @brief Determine the launch configuration for a device function for the maximum potential occupancy.
		 * @param Block The dimension of block.
		 * @tparam Vec The vector type.
		 * @tparam Func The device function type.
		 * @param func The device function symbol.
		 * @param threadSize The dimension of the thread.
		 * @param dynamicSMemSize Per-block dynamic shared memory usage intended, in bytes.
		 * @param blockSizeLimit The maximum block size func is designed to work with. 0 means no limit.
		 * @return The launch configurations
		*/
		template<STPDimensionSize Block, class Vec, class Func>
		__host__ STPLaunchConfiguration determineLaunchConfiguration(Func, const Vec&, size_t = 0u, int = 0);
#endif//__CUDACC_RTC__

//Hide device functions from the host compiler who cannot compile.
#ifdef __CUDACC__
		/**
		 * @brief Calculate the thread index in the device function of the current worker thread.
		 * @param Dim Specifies the dimension of the thread index to be calculated.
		 * @return The thread index of the current thread.
		 * This will be a tuple of thread indices in various dimension. Thread indices are ordered as (x, y, z).
		*/
		template<STPDimensionSize Dim>
		__device__ auto calcThreadIndex();
#endif

	}

}
#include "STPDeviceLaunchSetup.inl"
#endif//_STP_DEVICE_LAUNCH_SETUP_CUH_