#pragma once
#ifndef _STP_SINGLE_HISTOGRAM_FILTER_H_
#define _STP_SINGLE_HISTOGRAM_FILTER_H_

#include <SuperAlgorithm+/STPAlgorithmDefine.h>
#include <SuperTerrain+/World/Diversity/STPBiomeDefine.h>
//Engine Components
#include <SuperTerrain+/World/Chunk/STPNearestNeighbourInformation.hpp>
#include <SuperTerrain+/Utility/STPThreadPool.h>
//Single Histogram Data Structure
#include "STPSingleHistogram.hpp"

#include <utility>
#include <memory>

//GLM
#include <glm/vec2.hpp>

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
	public:

		/**
		 * @brief A STPFilterBuffer acts as an opaque memory unit for holding
		 * intermediate filter result and the final output for the single histogram filter.
		*/
		class STP_ALGORITHM_HOST_API STPFilterBuffer {
		private:

			friend class STPSingleHistogramFilter;

			/**
			 * @brief An opaque pointer to the internal data of the filter buffer.
			*/
			struct STPBufferMemory;
			std::unique_ptr<STPBufferMemory> Memory;

		public:

			//The number of bin and offset in the current histogram memory, respectively.
			typedef std::pair<size_t, size_t> STPHistogramSize;

			/**
			 * @brief Create a new filter buffer.
			 * The created buffer contains is not bounded to any particular histogram filter instance.
			 * It's recommended to reuse the buffer to avoid duplicate memory reallocation, if repeated execution to the filter is required.
			*/
			STPFilterBuffer();

			STPFilterBuffer(const STPFilterBuffer&) = delete;

			STPFilterBuffer(STPFilterBuffer&&) noexcept;

			STPFilterBuffer& operator=(const STPFilterBuffer&) = delete;

			STPFilterBuffer& operator=(STPFilterBuffer&&) noexcept;

			~STPFilterBuffer();

			/**
			 * @brief Retrieve the underlying contents in the filter buffer and pass them as pointers in STPSingleHistogram.
			 * @return The raw pointer to the histogram buffer. Note that the memory is bound to the current filter buffer and
			 * is only available while the calling instance is valid.
			 * @see STPSingleHistogram
			*/
			STPSingleHistogram readHistogram() const noexcept;

			/**
			 * @brief Get the size of the bin and offset.
			 * @param input_dim The dimension of one input samplemap on a single chunk.
			 * It is undefined behaviour if this dimension goes out of bound of the result,
			 * this will happen if the dimension is different from the dimension parameter supplied when getting the filter result.
			 * @return The histogram size information.
			*/
			STPHistogramSize size(const glm::uvec2&) const noexcept;

		};

	private:

		//A multi-thread worker for concurrent per-pixel histogram generation
		STPThreadPool FilterWorker;

		/**
		 * @brief Run the filter in parallel.
		 * @param sample_map The input sample map for filter.
		 * @param nn_info The information about the nearest_neighbour logic applies to the samplemap.
		 * @param filter_memory The filter buffer that will be for running the single histogram filter, and also output the final output.
		 * @param central_chunk_index The local coordinate points to the central chunk within the range of nearest neighbour chunks.
		 * @param radius The radius of the filter.
		*/
		void filterDistributed(const STPDiversity::Sample*, const STPNearestNeighbourInformation&, STPFilterBuffer::STPBufferMemory&, glm::uvec2, unsigned int);

	public:

		STPSingleHistogramFilter();

		STPSingleHistogramFilter(const STPSingleHistogramFilter&) = delete;

		STPSingleHistogramFilter(STPSingleHistogramFilter&&) = delete;

		STPSingleHistogramFilter& operator=(const STPSingleHistogramFilter&) = delete;

		STPSingleHistogramFilter& operator=(STPSingleHistogramFilter&&) = delete;

		~STPSingleHistogramFilter() = default;

		/**
		 * @brief Perform histogram filter on the input texture.
		 * @param samplemap The input samplemap. This is typically a merged nearest-neighbour samplemap.
		 * The input texture must be aligned in row-major order, and must be a available on host memory space.
		 * @param nn_info The information about the nearest-neighbour logic used by the samplemap.
		 * @param filter_buffer The filter buffer for execution of the filter and storing the final result.
		 * @param radius The filter radius
		 * @return The raw pointer resultant histogram of the execution.
		 * Note that the memory stored in output histogram is managed by the pointer provided in histogram_output.
		 * The same output can be retrieved later from the input filter buffer.
		 * @see STPFilterBuffer
		*/
		STPSingleHistogram operator()(const STPDiversity::Sample*, const STPNearestNeighbourInformation&, STPFilterBuffer&, unsigned int);

	};

}
#endif//_STP_SINGLE_HISTOGRAM_FILTER_H_