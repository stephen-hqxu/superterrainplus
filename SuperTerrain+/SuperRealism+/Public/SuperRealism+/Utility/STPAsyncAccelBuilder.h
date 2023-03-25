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

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPAsyncAccelBuilder is a simple utility for asynchronous acceleration structure building for ray tracing
	 * with a built-in support for internal double buffering.
	 * Multiple instances of such builder can be used together to construct a multi-level hierarchy.
	*/
	class STP_REALISM_API STPAsyncAccelBuilder {
	private:

		//front and back buffer
		typedef std::pair<STPSmartDeviceMemory::STPStreamedDevice<unsigned char[]>, OptixTraversableHandle> STPAccelStructBuffer;

		STPAccelStructBuffer AccelStruct[2];
		//pointer to each member in the acceleration structure buffer.
		const STPAccelStructBuffer* FrontBuffer;
		STPAccelStructBuffer* BackBuffer;

	public:

		/**
		 * @brief Holds parameter set for acceleration structure build operation.
		*/
		struct STPBuildInformation {
		public:

			//The device context where the build happens.
			OptixDeviceContext Context;
			//The stream where the build and memory operation happens.
			cudaStream_t Stream;
			//Accel options.
			const OptixAccelBuildOptions* AccelOptions;
			//An array of OptixBuildInput objects.
			const OptixBuildInput* BuildInputs;
			//Must be >= 1 for GAS, and == 1 for IAS.
			unsigned int numBuildInputs;
			//A temporary buffer.
			//The memory of temporary buffer should be managed by the user,
			//to allow maximum effectiveness and efficiency when building multiple level of AS by sharing the temporary buffer in the same stream.
			CUdeviceptr TempBuffer;
			//In bytes, the total size of the temporary buffer.
			size_t TempBufferSize;
			//In bytes, the total size of the output buffer.
			//This memory is managed internally using the double buffering mechanism.
			size_t OutputBufferSize;
			//Types of requested properties and output buffers.
			const OptixAccelEmitDesc* EmittedProperties = nullptr;
			//Number of post-build properties to populate (may be zero).
			unsigned int numEmittedProperties = 0u;

		};

		/**
		 * @brief Holds parameter set for acceleration structure compact operation.
		*/
		struct STPCompactInformation {
		public:

			//The device context where the compact happens.
			OptixDeviceContext Context;
			//The stream where the build and memory operation happens.
			cudaStream_t Stream;
			//The size of output buffer in byte, should be queried from emitted properties during AS build.
			size_t OutputBufferSize;

		};

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
		 * It only submits build event to the supplied stream and does not do any synchronisation.
		 * @param buildInfo Information about the build.
		 * @param memPool The memory pool from which memory is coming from and returned to.
		 * @return The traversable handle returned from the build query, whose memory is managed automatically.
		 * This is also the handle in the back buffer.
		*/
		OptixTraversableHandle build(const STPBuildInformation&, cudaMemPool_t);

		/**
		 * @brief Start a compact operation in the back buffer.
		 * It will operate on the back buffer; since it does not do any implicit synchronisation,
		 * it is strongly advised to put build and compact operation of the same AS in the same stream.
		 * For the best performance as advised by OptiX programming guide, it is a good idea to perform build and compact in batch,
		 * such as making these operations of same-level AS parallel.
		 * @param compactInfo Information about the compaction.
		 * @param memPool The memory pool from which memory is coming from and returned to.
		 * @return The traversable handle returned from the compact query, this will replaced the old handle in the back buffer.
		*/
		OptixTraversableHandle compact(const STPCompactInformation&, cudaMemPool_t);

		/**
		 * @brief Swap the front and back acceleration structure memory.
		 * This function does NOT check if the recent build event has finished.
		*/
		void swapHandle() noexcept;

		/**
		 * @brief Get the traversable handle of the acceleration structure in the front buffer.
		 * @return The traversable handle. If front buffer is not available, null is returned.
		*/
		OptixTraversableHandle getTraversableHandle() const noexcept;

	};

}
#endif//_STP_ASYNC_ACCEL_BUILDER_H_