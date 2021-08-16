#pragma once
#ifndef _STP_SINGLE_HISTOGRAM_FILTER_H_
#define _STP_SINGLE_HISTOGRAM_FILTER_H_

#include <SuperAlgorithm+/STPAlgorithmDefine.h>
//System
#include <vector>
//Engine Components
#include <Utility/STPThreadPool.h>
#include <World/Diversity/STPBiomeDefine.h>
#include <GPGPU/STPFreeSlipGenerator.cuh>
//GLM
#include <glm/vec2.hpp>

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
		 * @brief STPSingleHistogramFilter is an analysis tool for biomemap.
		 * It generates histogram for every pixel on the biomemap within a given radius.
		 * The bin or bucket size of the histogram will always be one, which denotes by the biomemap format "Sample".
		 * STPSingleHistogramFilter optimises for performant CPU computation, such that all memory provided to the histogram filter should be available on host side.
		 * An example use case is biome-edge interpolation, and space-partitioning biomes within a given radius to calculate a "factor" for linear interpolation.
		*/
		class STPALGORITHMPLUS_HOST_API STPSingleHistogramFilter {
		public:

			/**
			 * @brief Contains information about a bin.
			 * There is only one entry for each bin, such that each one uniquely represent an item, as well as the number of item presented.
			*/
			struct STPBin {
			public:

				//The item the bin is current holding
				STPDiversity::Sample Item;
				//Data for this item
				union {
				private:

					friend class STPSingleHistogramFilter;

					//The number of item the bin contains
					unsigned int Quantity;

				public:

					//The normalised weight of this item, it's the count divided by the sum of count over all items in this histogram.
					float Weight;

				} Data;

			};

			/**
			 * @brief Contains information of a histogram of one pixel.
			*/
			struct STPHistogram {
			public:

				//The number of bin in this histogram for this pixel
				unsigned int BinSize;
				//The index of STPBin from the beginning of the linear array of the texture per-pixel histogram to reach the current pixel
				unsigned int HistogramStartOffset;

			};

			/**
			 * @brief STPFilterReport contains the output of the result from running STPSingleHistogramFilter for the entire texture.
			 * Each pixel has one histogram, each histogram has some numebr of bins.
			 * All Bins are arranged in a contiguous linear memory, to get the bin for a pixel, BinStartOffset needs to be retrieved
			*/
			struct STPFilterReport {
			public:

				//Histogram for all pixels.
				STPHistogram* Histogram;
				//All bins extracted from histogram, it's a flatten array of histograms for every pixel.
				//The bins of the next histogram is connected to that of the previous histogram, such that memory is contiguous.
				STPBin* Bin;

			};

		private:

			/**
			 * @brief STPHistogramBuffer resembles STPHistogram, unlike which, this is a compute buffer during generation instead of a data structure that
			 * can be easily used by external environment directly.
			*/
			struct STPHistogramBuffer;

			/**
			 * @brief Accumulator acts as a cache for each row or column iteration.
			 * The accumulator for the next pixel equals to the accumulator from the previous pixel plus the left/down out-of-range radius pixel and
			 * minux the right/up in-range radius pixel.
			*/
			class STPAccumulator;

			//After some experiment, we found out 4 parallel workers is the sweet spot.
			constexpr static unsigned char DEGREE_OF_PARALLELISM = 4u;

			//A multi-thread worker for concurrent per-pixel histogram generation
			STPThreadPool filter_worker;

			//Each worker will be assigned a cache, and join them together when synced.
			std::unique_ptr<STPHistogramBuffer[]> Cache;
			std::unique_ptr<STPAccumulator[]> Accumulator;
			//The output bins from each phase of computation, it's a flat array of STPBin that contains bins for every pixel
			std::unique_ptr<STPHistogramBuffer> Output;

			/**
			 * @brief Perform horizontal pass histogram filter
			 * @param sample_map The input sample map, usually it's biomemap
			 * @param h_range Denotes the height start and end that will be computed by the current function call.
			 * The range should start from the halo (central image y index minus radius), and should use global index.
			 * The range end applies as well (central image y index plus dimension plus radius)
			 * @param threadID the ID of the CPU thread that is calling this function
			 * @param radius The radius of the filter.
			*/
			void filter_horizontal(const STPFreeSlipSampleManager&, glm::uvec2, unsigned char, unsigned int);

		public:

			/**
			 * @brief Init single histogram filter.
			*/
			STPSingleHistogramFilter();

			/**
			 * @brief Init single histogram filter with hints.
			 * Provide 0 or empty-initialised value if hint is unknown.
			 * @param dimension_hint The maximum dimension of the texture that will be used.
			 * This acts only as an estimation such that the program can pre-allocate size of internal memory pool.
			 * @param radius_hint The maximum radius of the filter that is going to be run.
			 * This acts as an estimation similar to dimension_hint, if multiple radius is planned to used, provide the largest number for best performance.
			 * @param max_sample_hint A hint that the max value "sample" can go in any biomemap
			 * @param partition_hint A theoretically possible number of biomes that can present in a radius
			*/
			STPSingleHistogramFilter(glm::uvec2, unsigned int, STPDiversity::Sample, unsigned int);

			~STPSingleHistogramFilter() = default;

			STPSingleHistogramFilter(const STPSingleHistogramFilter&) = delete;

			STPSingleHistogramFilter(STPSingleHistogramFilter&&) = delete;

			STPSingleHistogramFilter& operator=(const STPSingleHistogramFilter&) = delete;

			STPSingleHistogramFilter& operator=(STPSingleHistogramFilter&&) = delete;

		};

	}
}
#endif//_STP_SINGLE_HISTOGRAM_FILTER_H_