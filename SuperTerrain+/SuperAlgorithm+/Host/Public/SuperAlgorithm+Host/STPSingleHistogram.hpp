#pragma once
#ifndef _STP_SINGLE_HISTOGRAM_HPP_
#define _STP_SINGLE_HISTOGRAM_HPP_

//Engine Component
#include <SuperTerrain+/World/STPWorldMapPixelFormat.hpp>

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPSingleHistogram contains the output of the result from running STPSingleHistogramFilter for the entire texture.
	 * Each pixel has one histogram, each histogram has some number of bins.
	 * All Bins are arranged in a contiguous linear memory, to get the bin for a pixel, BinStartOffset needs to be retrieved.
	*/
	struct STPSingleHistogram {
	public:

		/**
		 * @brief Contains information about a bin.
		 * There is only one entry for each bin, such that each one uniquely represent an item, as well as the number of item presented.
		 * @tparam WT The type of the bin weight.
		*/
		template<typename WT>
		struct STPGenericBin {
		public:

			//The item the bin is current holding
			STPSample_t Item;
			//The weight of this bin.
			//Use a floating point type to denote a normalised weight of this item,
			//	it's the count divided by the sum of count over all items in this histogram.
			//Use a integral type for un-normalised weight, i.e., The number of item the bin contains.
			//	Use of a un-normalised weight is only used during generation of single histogram,
			//	and the end-user should stick to normalised weight.
			WT Weight;

		};
		//The bin that uses normalised weight.
		typedef STPGenericBin<float> STPBin;

		//All bins extracted from histogram, it's a flatten array of histograms for every pixel.
		//The bins of the next histogram is connected to that of the previous histogram, such that memory is contiguous.
		//The number of element this array contains is the number read from the last element in HistogramStartOffset.
		const STPBin* Bin;
		//The index of STPBin from the beginning of the linear array of the texture per-pixel histogram to reach the
		//current pixel The number of element in this array is the same as the dimension (of one texture) in the input,
		//plus one element at the end denotes the total size of the bin
		const unsigned int* HistogramStartOffset;

	};

}
#endif//_STP_SINGLE_HISTOGRAM_HPP_