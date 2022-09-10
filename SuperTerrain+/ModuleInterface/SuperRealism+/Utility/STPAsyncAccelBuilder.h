#pragma once
#ifndef _STP_ASYNC_ACCEL_BUILDER_H_
#define _STP_ASYNC_ACCEL_BUILDER_H_

#include <SuperRealism+/STPRealismDefine.h>
//Memory
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceMemory.h>

//CUDA
#include <optix_types.h>
#include <cuda_runtime.h>

//System
#include <utility>
#include <atomic>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPAsyncAccelBuilder is a simple utility for asynchronous acceleration structure building for ray tracing
	 * with a built-in support for internal buffering.
	*/
	class STP_REALISM_API STPAsyncAccelBuilder {
	private:

		//front and back buffer
		typedef std::pair<STPSmartDeviceMemory::STPStreamedDeviceMemory<unsigned char[]>, OptixTraversableHandle> STPAccelStructBuffer;

		STPAccelStructBuffer AccelStruct[2];
		//pointer to each member in the acceleration structure buffer.
		const STPAccelStructBuffer* FrontBuffer;
		STPAccelStructBuffer* BackBuffer;

		//Status flags for acceleration structure build operation.
		struct STPBuildStatus {
		public:

			//Indicate if there is a build operation in progress, and has the last finished build operation swapped.
			std::atomic_bool IsBackBufferBusy, HasPendingSwap;

		} AccelStatusFlag;

	public:

		/**
		 * @brief Initialise an instance of asynchronous AS builder.
		 * Initially all internal memory are empty.
		*/
		STPAsyncAccelBuilder();

		STPAsyncAccelBuilder(const STPAsyncAccelBuilder&) = delete;

		STPAsyncAccelBuilder(STPAsyncAccelBuilder&&) = delete;

		STPAsyncAccelBuilder& operator=(const STPAsyncAccelBuilder&) = delete;

		STPAsyncAccelBuilder& operator=(STPAsyncAccelBuilder&&) = delete;

		~STPAsyncAccelBuilder() = default;

		/**
		 * @brief Start a build operation in the back buffer.
		 * Build operation will not happen if there is a pending build going on in the back buffer.
		 * @param context The device context where the build happens.
		 * @param stream The stream where the build and memory operation happens.
		 * @param memPool The memory pool from which memory is coming from and returned to.
		 * @param accelOptions Accel options.
		 * @param buildInputs An array of OptixBuildInput objects.
		 * @param numBuildInputs Must be >= 1 for GAS, and == 1 for IAS.
		 * @param tempBuffer A temporary buffer.
		 * The memory of temporary buffer should be managed by the user,
		 * to allow maximum effectiveness and efficiency when building multiple level of AS by sharing the temporary buffer in the same stream.
		 * @param tempBufferSize In bytes, the total size of the temporary buffer.
		 * @param outputBufferSize In bytes, the total size of the output buffer.
		 * This memory is managed internally using the double buffering mechanism.
		 * @param emittedProperties Types of requested properties and output buffers.
		 * @param numEmittedProperties number of post-build properties to populate (may be zero).
		 * @return A status indicating if build requested has been submitted.
		*/
		bool build(OptixDeviceContext, cudaStream_t, cudaMemPool_t, const OptixAccelBuildOptions&,
			const OptixBuildInput*, unsigned int, CUdeviceptr, size_t, size_t, const OptixAccelEmitDesc* = nullptr, unsigned int = 0u);

		/**
		 * @brief Swap the front and back acceleration structure memory.
		 * Swap will happen only when back buffer is not busy, and there is a recently finished and un-swapped buffer.
		 * @return A status flag indicating if swap operation is done.
		*/
		bool swapHandle();

		/**
		 * @brief Get the traversable handle of the acceleration structure in the front buffer.
		 * @return The traversable handle. If front buffer is not available, null is returned.
		*/
		OptixTraversableHandle getTraversableHandle() const;

	};

}
#endif//_STP_ASYNC_ACCEL_BUILDER_H_