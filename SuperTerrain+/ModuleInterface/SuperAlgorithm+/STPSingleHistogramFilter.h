#pragma once
#ifndef _STP_SINGLE_HISTOGRAM_FILTER_H_
#define _STP_SINGLE_HISTOGRAM_FILTER_H_

#include <SuperAlgorithm+/STPAlgorithmDefine.h>
#include <SuperTerrain+/World/Diversity/STPBiomeDefine.h>
//Engine Components
#include <SuperTerrain+/World/Chunk/STPFreeSlipInformation.hpp>
//Single Histogram Data Structure
#include "STPSingleHistogram.hpp"

#include <memory>

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPSingleHistogramFilter is an analysis tool for discrete format texture such as the biomemap.
	 * It generates histogram for every pixel on the texture within a given radius.
	 * The bin or bucket size of the histogram will always be one, which denotes by the biomemap format "Sample".
	 * An example use case is biome-edge interpolation, and space-partitioning biomes within a given radius to calculate a "factor" for linear interpolation.
	 * 
	 * STPSingleHistogramFilter optimises for efficient multi-thread CPU computation,
	 * with careful manual tuning and algorithm design, the single histogram filter achieves a asymptotic runtime of constant with respect to the radius of the filter kernel.
	 * Strictly speaking, the worst case runtime complexity is `O(N + r)` where *N* is the total number of pixel on the texture and *r* is the radius of kernel.
	 * In reality, *r* is negligibly small compared to *N*.
	 * 
	 * The filter also contains an internal adaptive memory pool that serves as a cache during computation, the first few executions will be slower due to the first-time allocation,
	 * but once reused for repetitive filter calls little to no memory allocation should happen and performance will go to summit.
	*/
	class STP_ALGORITHM_HOST_API STPSingleHistogramFilter {
	private:

		/**
		 * @brief STPHistogramBuffer resembles STPHistogram, unlike which, this is a compute buffer during generation instead of a data structure that
		 * can be easily used by external environment directly.
		 * @tparam Pinned True to use pinned memory allocator for the histogram buffer
		*/
		template<bool Pinned>
		struct STPHistogramBuffer;

		typedef STPHistogramBuffer<false> STPDefaultHistogramBuffer;
		typedef STPHistogramBuffer<true> STPPinnedHistogramBuffer;

		/**
		 * @brief The default membered deleter for pinned histogram buffer that will be passed to external user
		*/
		struct STP_ALGORITHM_HOST_API STPPinnedHistogramBufferDeleter {
		public:

			void operator()(STPPinnedHistogramBuffer*) const;

		};

		/**
		 * @brief STPSHFKernel is the implementation of the single histogram filter.
		*/
		class STPSHFKernel;
		std::unique_ptr<STPSHFKernel> Kernel;

	public:

		/**
		 * @brief An opaque pointer to the type STPHistogramBuffer.
		 * Filter result from running the filter will be stored in the object.
		 * Direct access to the underlying histogram is not available, but can be retrieved.
		 * @see STPHistogramBuffer
		*/
		typedef std::unique_ptr<STPPinnedHistogramBuffer, STPPinnedHistogramBufferDeleter> STPHistogramBuffer_t;

		/**
		 * @brief Init single histogram filter.
		*/
		STPSingleHistogramFilter();

		//The destructor is default, but since we are using unique_ptr to some incomplete nested types, we need to hide the destructor.
		~STPSingleHistogramFilter();

		STPSingleHistogramFilter(const STPSingleHistogramFilter&) = delete;

		STPSingleHistogramFilter(STPSingleHistogramFilter&&) = delete;

		STPSingleHistogramFilter& operator=(const STPSingleHistogramFilter&) = delete;

		STPSingleHistogramFilter& operator=(STPSingleHistogramFilter&&) = delete;

		/**
		 * @brief Create a histogram buffer which holds the histogram after the filter execution.
		 * The created histogram buffer contains a memory pool as well, and it's not bounded to any particular histogram filter instance.
		 * It's recommended to reuse the buffer as well to avoid duplicate memory reallocation
		 * @return The smart pointer to the histogram buffer. The buffer is managed by unique_ptr and will be freed automatically when destroyed
		*/
		static STPHistogramBuffer_t createHistogramBuffer();

		/**
		 * @brief Perform histogram filter on the input texture.
		 * If there is a histogram returned and no destroyHistogram() is called, execution is thrown and no execution is launched.
		 * @param samplemap The input free-slip manager with sample_map loaded
		 * The input texture must be aligned in row-major order, and must be a available on host memory space.
		 * @param freeslip_info The information about the free-slip logic.
		 * @param histogram_output The histogram buffer where the final output will be stored
		 * @param radius The filter radius
		 * @return The raw pointer resultant histogram of the execution.
		 * Note that the memory stored in output histogram is managed by the pointer provided in histogram_output.
		 * The same output can be retrieved later by calling function readHistogramBuffer()
		 * @see readHistogramBuffer()
		*/
		STPSingleHistogram operator()(const STPDiversity::Sample*, const STPFreeSlipInformation&, const STPHistogramBuffer_t&, unsigned int);

		/**
		 * @brief Retrieve the underlying contents in the histogram buffer and pass them as pointers in STPSingleHistogram
		 * @param buffer The pointer to the histogram buffer
		 * @return The raw pointer to the histogram buffer. Note that the memory is bound to the buffer provided and
		 * is only available while STPHistogramBuffer_t is valid.
		 * @see STPSingleHistogram
		*/
		static STPSingleHistogram readHistogramBuffer(const STPHistogramBuffer_t&);

	};

}
#endif//_STP_SINGLE_HISTOGRAM_FILTER_H_