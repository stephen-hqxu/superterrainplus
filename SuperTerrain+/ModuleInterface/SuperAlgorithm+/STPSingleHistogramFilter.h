#pragma once
#ifndef _STP_SINGLE_HISTOGRAM_FILTER_H_
#define _STP_SINGLE_HISTOGRAM_FILTER_H_

#include <SuperAlgorithm+/STPAlgorithmDefine.h>
//Container
#include <queue>
#include <list>
#include <array>
#include <utility>
//Thread Safety
#include <mutex>

//Engine Components
#include <SuperTerrain+/Utility/STPThreadPool.h>
#include <SuperTerrain+/World/Chunk/FreeSlip/STPFreeSlipManager.cuh>
//GLM
#include <glm/vec2.hpp>
//Single Histogram Data Structure
#include "STPSingleHistogram.hpp"

namespace SuperTerrainPlus::STPCompute {

	/**
	 * @brief STPSingleHistogramFilter is an analysis tool for biomemap.
	 * It generates histogram for every pixel on the biomemap within a given radius.
	 * The bin or bucket size of the histogram will always be one, which denotes by the biomemap format "Sample".
	 * An example use case is biome-edge interpolation, and space-partitioning biomes within a given radius to calculate a "factor" for linear interpolation.
	 * STPSingleHistogramFilter optimises for performant CPU computation, such that all memory provided to the histogram filter should be available on host side.
	 * The filter also contains an internal adaptive memory pool that serves as a cache during computation, the first few executions will be slower due to the first-time allocation,
	 * but once reused for repetitive filter calls little to no memory allocation should happen and performance will go to summit.
	 * so memory can be reused and no re-allocation is required.
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
		 * @brief Accumulator acts as a cache for each row or column iteration.
		 * The accumulator for the next pixel equals to the accumulator from the previous pixel plus the left/down out-of-range radius pixel and
		 * minux the right/up in-range radius pixel.
		*/
		class STPAccumulator;

		//After some experiment, we found out 4 parallel workers is the sweet spot.
		constexpr static unsigned char Parallelism = 4u;
		//A multi-thread worker for concurrent per-pixel histogram generation
		STPThreadPool filter_worker;

		//A workplace is some available memory for a complete histogram generation
		typedef std::pair<STPDefaultHistogramBuffer, STPAccumulator> STPWorkplace;
		typedef std::unique_ptr<STPWorkplace[]> STPDepartment;
		typedef std::list<STPDepartment> STPOrganisation;
		typedef STPOrganisation::iterator STPOrganisation_it;
		//All available workplace.
		//list of pointers
		STPOrganisation Organisation;
		std::queue<STPOrganisation_it, std::list<STPOrganisation_it>> FreeWorkingMemory;
		std::mutex WorkplaceLock;

		/**
		 * @brief Request an available workplace for workers to generate histogram.
		 * Workplace guarantees critical access, meaning all memory resides will not be modified by other workers until it is returned.
		 * @return The iterator to the free workplace.
		*/
		STPOrganisation_it requestWorkplace();

		/**
		 * @brief Return a workplace back to the system so it can be used by other tasks later.
		 * @param it The iterator of workplace to be returned.
		*/
		void returnWorkplace(STPOrganisation_it);

		/**
		 * @brief Copy the content in accumulator to the histogram buffer.
		 * Caller should make sure Output buffer has been preallocated, the size equals to the sum of all thread buffers.
		 * @param target The target histogram buffer
		 * @param acc The accumulator to be copied
		 * @param normalise True to normalise the histogram in accumulator before copying.
		 * After normalisation, STPBin.Data should use Weight rather than Quantity, and the sum of weight of all bins in the accumulator is 1.0f
		*/
		static void copy_to_buffer(STPDefaultHistogramBuffer&, STPAccumulator&, bool);

		/**
		 * @brief Perform vertical pass histogram filter
		 * @param sample_map The input sample map, usually it's biomemap
		 * @param vertical_start_offset The vertical starting offset on the texture.
		 * The start offset should make the worker starts at the first y coordinate of the central texture.
		 * @param w_range Denotes the width start and end that will be computed by the current function call.
		 * The range should start from the halo (central image x index minus radius), and should use global index.
		 * The range end applies as well (central image x index plus dimension plus radius)
		 * @param workplace The pointer to the allocated working memory.
		 * @param radius The radius of the filter.
		*/
		void filter_vertical(const STPFreeSlipSampleManager&, unsigned int, glm::uvec2, STPWorkplace&, unsigned int);

		/**
		 * @brief Merge buffers from each thread into a large chunk of output data.
		 * It will perform offset correction for HistogramStartOffset.
		 * @param buffer The histogram buffer that will be merged to
		 * @param workplace_memory The pointer to the thread memory where the buffer will be copied from.
		 * @param workplaceID The ID of the workplace in the department. Note that threadID 0 doesn't require offset correction.
		 * @param output_base The base start index from the beginning of output container for each thread for bin and histogram offset
		*/
		void copy_to_output(STPPinnedHistogramBuffer*, const STPDefaultHistogramBuffer&, unsigned char, glm::uvec2);

		/**
		 * @brief Perform horizontal pass histogram filter.
		 * The input is the ouput from horizontal pass
		 * @param histogram_input The output histogram buffer from the vertical pass
		 * @param dimension The dimension of one texture
		 * @param h_range Denotes the height start and end that will be computed by the current function call.
		 * The range should start from 0.
		 * The range end at the height of the texture
		 * @param workplace The pointer to the allocated working memory.
		 * @param radius The radius of the filter
		*/
		void filter_horizontal(STPPinnedHistogramBuffer*, const glm::uvec2&, glm::uvec2, STPWorkplace&, unsigned int);

		/**
		 * @brief Performa a complete histogram filter
		 * @param sample_map The input sample map for filter.
		 * @param histogram_output The histogram buffer that will be used as buffer, and also output the final output
		 * @param central_chunk_index The local free-slip coordinate points to the central chunk.
		 * @param radius The radius of the filter
		*/
		void filter(const STPFreeSlipSampleManager&, STPPinnedHistogramBuffer*, glm::uvec2, unsigned int);

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
		 * @param samplemap_manager The input free-slip manager with sample_map loaded
		 * The input texture must be aligned in row-major order, and must be a host manager
		 * @param histogram_output The histogram buffer where the final output will be stored
		 * @param radius The filter radius
		 * @return The raw pointer resultant histogram of the execution.
		 * Note that the memory stored in output histogram is managed by the pointer provided in histogram_output.
		 * The same output can be retrieved later by calling function readHistogramBuffer()
		 * @see readHistogramBuffer()
		*/
		STPSingleHistogram operator()(const STPFreeSlipSampleManager&, const STPHistogramBuffer_t&, unsigned int);

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