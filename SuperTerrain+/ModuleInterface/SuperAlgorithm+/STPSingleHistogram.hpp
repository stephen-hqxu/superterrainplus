#pragma once
#ifndef _STP_SINGLE_HISTOGRAM_HPP_
#define _STP_SINGLE_HISTOGRAM_HPP_

//Engine Component
#include <World/Diversity/STPBiomeDefine.h>

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
		 * @brief STPSingleHistogram contains the output of the result from running STPSingleHistogramFilter for the entire texture.
		 * Each pixel has one histogram, each histogram has some numebr of bins.
		 * All Bins are arranged in a contiguous linear memory, to get the bin for a pixel, BinStartOffset needs to be retrieved
		*/
		struct STPSingleHistogram {
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

			//All bins extracted from histogram, it's a flatten array of histograms for every pixel.
			//The bins of the next histogram is connected to that of the previous histogram, such that memory is contiguous.
			//The number of element this array contains is the number read from the last element in HistogramStartOffset.
			const STPBin* Bin;
			//The index of STPBin from the beginning of the linear array of the texture per-pixel histogram to reach the current pixel
			//The number of element in this array is the same as the dimension (of one texture) in the input, plus one element at the end denotes the total size of the bin
			const unsigned int* HistogramStartOffset;

		};

	}
}
#endif//_STP_SINGLE_HISTOGRAM_HPP_