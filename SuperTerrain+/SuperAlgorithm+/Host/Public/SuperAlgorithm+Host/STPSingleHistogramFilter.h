#pragma once
#ifndef _STP_SINGLE_HISTOGRAM_FILTER_H_
#define _STP_SINGLE_HISTOGRAM_FILTER_H_

#include <SuperAlgorithm+Host/STPAlgorithmDefine.h>
#include <SuperTerrain+/World/STPWorldMapPixelFormat.hpp>
//Engine Components
#include <SuperTerrain+/World/Chunk/STPNearestNeighbourInformation.hpp>
#include <SuperTerrain+/Utility/STPThreadPool.h>
//Single Histogram Data Structure
#include "STPSingleHistogram.hpp"

#include <utility>
#include <memory>
#include <variant>

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
		public:

			/**
			 * @brief STPExecutionType specifies the type of execution where the filter buffer is intended for.
			*/
			enum class STPExecutionType : unsigned char {
				Serial = 0x00u,
				Parallel = 0xFFu
			};

		private:

			friend class STPSingleHistogramFilter;

			/**
			 * @brief An opaque pointer to the internal data of the filter buffer.
			 * @param Exec Specifies the type of execution the buffer is used for.
			*/
			template<STPExecutionType Exec>
			struct STPBufferMemory;

			//serial version of buffer memory
			typedef STPBufferMemory<STPExecutionType::Serial> STPSerialBufferMemory;
			//parallel version of buffer memory
			typedef STPBufferMemory<STPExecutionType::Parallel> STPParallelBufferMemory;

			//The memory holding the intermediate filter cache and the filter result, for single and multithreaded execution.
			typedef std::variant<std::unique_ptr<STPParallelBufferMemory>, std::unique_ptr<STPSerialBufferMemory>> STPStrategicBufferMemory;
			STPStrategicBufferMemory Memory;

			/**
			 * @brief Get the output histogram in the buffer memory.
			 * @return The pointer to the output histogram.
			*/
			const auto& getOutput() const;

		public:

			//The number of bin and offset in the current histogram memory, respectively.
			typedef std::pair<size_t, size_t> STPHistogramSize;

			/**
			 * @brief Create a new filter buffer.
			 * The created buffer contains is not bounded to any particular histogram filter instance.
			 * It's recommended to reuse the buffer to avoid duplicate memory reallocation, if repeated execution to the filter is required.
			 * @param execution_type Specifies type of execution.
			*/
			STPFilterBuffer(STPFilterBuffer::STPExecutionType);

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
			STPSingleHistogram readHistogram() const;

			/**
			 * @brief Get the size of the bin and offset.
			 * @return The histogram size information.
			*/
			STPHistogramSize size() const;

			/**
			 * @brief Get the execution type of current filter buffer that is specialised for.
			 * @return The execution type of the current filter buffer.
			*/
			STPFilterBuffer::STPExecutionType type() const noexcept;

		};

	private:

		/**
		 * @brief STPFilterKernelData stored arguments required for the filter.
		*/
		struct STPFilterKernelData {
		public:

			//The input sample map for filter.
			const STPSample_t* const SampleMap;
			//The information about the nearest_neighbour logic applies to the sample-map.
			const STPNearestNeighbourInformation& NeighbourInfo;
			//The filter buffer that will be for running the single histogram filter, and also output the final output.
			STPFilterBuffer& FilterMemory;
			//The radius of the filter.
			const unsigned int Radius;

			//The coordinate of the first (top-left corner) pixel on the filtering sub-texture for the vertical pass.
			const unsigned int VerticalStartingCoord;
			//The X and Y range of the texture to be filtered during the vertical and horizontal pass.
			const glm::uvec2 WidthRange, HeightRange;

		};

		//A multi-thread worker for concurrent per-pixel histogram generation
		STPThreadPool FilterWorker;

		/**
		 * @brief Run the filter in single thread.
		 * @param data The filter kernel data.
		*/
		void filter(const STPFilterKernelData&);

		/**
		 * @brief Run the filter in parallel.
		 * @param data The filter kernel data.
		*/
		void filterDistributed(const STPFilterKernelData&);

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
		STPSingleHistogram operator()(const STPSample_t*, const STPNearestNeighbourInformation&, STPFilterBuffer&, unsigned int);

	};

}
#endif//_STP_SINGLE_HISTOGRAM_FILTER_H_