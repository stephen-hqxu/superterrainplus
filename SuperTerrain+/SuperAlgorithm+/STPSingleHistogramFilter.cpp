#pragma once
#include <SuperAlgorithm+/STPSingleHistogramFilter.h>

#include <limits>
#include <iterator>
#include <algorithm>

#include <glm/common.hpp>

using namespace SuperTerrainPlus::STPCompute;
using SuperTerrainPlus::STPDiversity::Sample;

using glm::vec2;
using glm::uvec2;

using std::vector;
using std::make_unique;
using std::pair;
using std::make_pair;

/* Private object implementations */

struct STPSingleHistogramFilter::STPHistogramBuffer {
public:

	//All flatten bins in all histograms
	vector<STPBin> Bin;
	//Get the bin starting index for a pixel in the flatten bin array
	vector<unsigned int> HistogramStartOffset;

	/**
	 * @brief Clear containers in histogram buffer.
	 * It doesn't gaurantee to free up memory allocated inside, acting as memory pool which can be reused.
	*/
	inline void clear() {
		this->Bin.clear();
		this->HistogramStartOffset.clear();
	}

};

class STPSingleHistogramFilter::STPAccumulator {
private:

	//Denotes there is no entry in the dictionary that corresponds to the index in bin
	constexpr static unsigned int NO_ENTRY = std::numeric_limits<unsigned int>::max();

	/**
	 * @brief Get the bin for a biome sample.
	 * If bin doesn't exist, a new bin is created.
	 * @param sample The biome sample
	 * @return The pointer to the biome bin
	*/
	STPBin& operator[](Sample sample) {
		//check if the sample is in the dictionary
		const int diff = static_cast<int>(this->Dictionary.size() - sample);
		if (diff <= 0) {
			//if not we need to insert that many extra entries so we can use sample to index the dictionary directly
			this->Dictionary.insert(this->Dictionary.end(), (-diff) + 1, NO_ENTRY);
		}

		//get the biome using the index from dictionary
		unsigned int& bin_index = this->Dictionary[sample];
		if (bin_index == NO_ENTRY) {
			//biome not exist, add and initialise
			STPBin& bin = this->Bin.emplace_back();
			bin.Item = sample;
			bin.Data.Quantity = 0u;
			//record the index in the bin and store to dictionary
			bin_index = this->Bin.size() - 1;
			return bin;
		}
		return this->Bin[bin_index];
	}

public:

	//Use Sample as index, find the index in Bin for this sample
	vector<unsigned int> Dictionary;
	//Store the number of element
	vector<STPBin> Bin;

	//Array of bins in accumulator
	typedef vector<STPBin>::const_iterator STPBinIterator;
	typedef pair<STPBinIterator, STPBinIterator> STPBinArray;

	/**
	 * @brief Increment the sample bin by count
	 * @param sample The sample bin that will be operated on
	 * @param count The number to increment
	*/
	inline void inc(Sample sample, unsigned int count) {
		(*this)[sample].Data.Quantity += count;
	}

	/**
	 * @brief Decrement the sample bin by count
	 * If the bin will become empty after decrementing, bin will be erased from the accumulator as well as internal dictionary.
	 * Erasure in vector is expensive, so don't call this function too often.
	 * @param sample The sample bin that will be operated on
	 * @param count The number to decrement
	*/
	void dec(Sample sample, unsigned int count){
		//our algorithm guarantees the bin has been increment by this sample before, so no check is needed
		unsigned int& bin_index = this->Dictionary[sample];
		STPBin& bin = this->Bin[static_cast<unsigned int>(bin_index)];
		unsigned int& quant = bin.Data.Quantity;

		if (quant <= count) {
			//bin will become empty, erase this bin and dictionary entry
			this->Bin.erase(this->Bin.begin() + static_cast<unsigned int>(bin_index));
			bin_index = NO_ENTRY;
			return;
		}
		//simply decrement the quantity otherwise
		quant -= count;
	}

	/**
	 * @brief Get the array of bins in the accumulator, this is useful for doing fast memory copy
	 * @return Number of element in the bin and the pointer to the start of the bin array
	*/
	inline STPBinArray operator()() const {
		return make_pair(this->Bin.cbegin(), this->Bin.cend());
	}

	/**
	 * @brief Clear all content in accumulator, but leave reserved memory
	*/
	inline void clear() {
		this->Dictionary.clear();
		this->Bin.clear();
	}

};

/* Histogram Filter implementation */

STPSingleHistogramFilter::STPSingleHistogramFilter() : filter_worker(STPSingleHistogramFilter::DEGREE_OF_PARALLELISM) {
	//Initialise pointer to implementations
	this->Cache = make_unique<STPHistogramBuffer[]>(STPSingleHistogramFilter::DEGREE_OF_PARALLELISM);
	this->Accumulator = make_unique<STPAccumulator[]>(STPSingleHistogramFilter::DEGREE_OF_PARALLELISM);
	this->Output = make_unique<STPHistogramBuffer>();
}

STPSingleHistogramFilter::STPSingleHistogramFilter(uvec2 dimension_hint, unsigned int radius_hint, Sample max_sample_hint, unsigned int partition_hint) : STPSingleHistogramFilter() {
	//In horizontal pass, we need to take the bottom and top halo, each of which has width of radius
	//In vertical pass, only the central area is needed
	const unsigned int pixel = dimension_hint.x * (dimension_hint.y + radius_hint * 2u),
		pixel_per_thread = glm::ceil(1.0f * pixel / STPSingleHistogramFilter::DEGREE_OF_PARALLELISM);

	//Preallocate room
	for (unsigned int i = 0; i < STPSingleHistogramFilter::DEGREE_OF_PARALLELISM; i++) {
		this->Cache[i].Bin.reserve(pixel_per_thread);
		this->Cache[i].HistogramStartOffset.reserve(pixel_per_thread);

		this->Accumulator[i].Dictionary.reserve(max_sample_hint);
		this->Accumulator[i].Bin.reserve(partition_hint);
	}
	this->Output->HistogramStartOffset.reserve(pixel);
	this->Output->Bin.reserve(pixel);
}

void STPSingleHistogramFilter::filter_horizontal(const STPFreeSlipSampleManager& sample_map, uvec2 h_range, unsigned char threadID, unsigned int radius) {
	STPHistogramBuffer& target = this->Cache[threadID];
	STPAccumulator& acc = this->Accumulator[threadID];
	//make sure both of them are cleared (don't deallocate)
	target.clear();
	acc.clear();
	auto copy_to_buffer = [&target](STPAccumulator::STPBinArray& acc_arr) -> void {
		//TODO: since HistogramStartOffset size is fixed, maybe use a static array and access each elemenet using variable ti ?
		target.HistogramStartOffset.emplace_back(static_cast<unsigned int>(target.Bin.size()));//prevous bin size
		std::copy(acc_arr.first, acc_arr.second, std::back_inserter(target.Bin));
	};

	//we assume the radius never goes out of the free-slip boundary
	for (unsigned int i = h_range.x; i < h_range.y; i++) {
		//the target (current) pixel index
		unsigned int ti = i * sample_map.Data->FreeSlipRange.x,
			//the pixel index of left-most radius
			li = ti - radius,
			//the pixel index of right-most radius
			ri = ti + radius;

		//load radius strip for the first pixel into accumulator
		//we exclude the right most pixel in the radius to avoid duplicate loading later.
		//since li is overlapped when loading to histogram later, we repeat that pixel once
		acc.inc(sample_map[li], 1u);
		for (int j = -static_cast<int>(radius); j <= static_cast<int>(radius); j++) {
			acc.inc(sample_map[ti + j], 1u);
		}
		//copy the first pixel radius to buffer
		copy_to_buffer(acc());
		//generate histogram, starting from the second pixel
		for (unsigned int j = 1u; j < sample_map.Data->Dimension.x; j++) {
			//load one pixel to the right while unloading one pixel from the left
			acc.inc(sample_map[ri++], 1u);
			acc.dec(sample_map[li++], 1u);
			
			//copy acc to buffer
			copy_to_buffer(acc());

			//advance to the next central pixel
			ti++;
		}

		//clear the accumulator before starting the next row
		acc.clear();
	}
}