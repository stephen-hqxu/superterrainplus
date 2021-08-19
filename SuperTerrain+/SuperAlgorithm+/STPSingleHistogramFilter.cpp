#pragma once
#include <SuperAlgorithm+/STPSingleHistogramFilter.h>

#include <limits>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <functional>

#include <stdexcept>

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

	STPHistogramBuffer() = default;

	STPHistogramBuffer(const STPHistogramBuffer&) = delete;

	STPHistogramBuffer(STPHistogramBuffer&&) = delete;

	STPHistogramBuffer& operator=(const STPHistogramBuffer&) = delete;

	STPHistogramBuffer& operator=(STPHistogramBuffer&&) = delete;

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

	STPAccumulator() = default;

	STPAccumulator(const STPAccumulator&) = delete;

	STPAccumulator(STPAccumulator&&) = delete;

	STPAccumulator& operator=(const STPAccumulator&) = delete;

	STPAccumulator& operator=(STPAccumulator&&) = delete;

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
	 * If the bin will become empty after decrementing, bin will be erased from the accumulator and dictionary will be rehashed.
	 * Those operations are expensive, so don't call this function too often.
	 * In our implementation erasure rarely happens, benchmark shows this is still the best method.
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
			const unsigned int followed_index = this->Bin.erase(this->Bin.begin() + bin_index) - this->Bin.begin();
			bin_index = NO_ENTRY;

			//update the dictionary entries linearly, basically it's a rehash in hash table
			for (unsigned int& dic_index : this->Dictionary) {
				//since all subsequent indices followed by the erased bin has been advanced forward by one block
				//we need to subtract the indices recorded in dictionary for those entries by one.
				if (dic_index == NO_ENTRY || dic_index <= followed_index) {
					continue;
				}
				dic_index--;
			}

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
	this->ReportInUsed = false;
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

void STPSingleHistogramFilter::copy_to_buffer(STPHistogramBuffer& target, STPAccumulator& acc, bool normalise) {
	STPAccumulator::STPBinArray acc_arr = acc();
	target.HistogramStartOffset.emplace_back(static_cast<unsigned int>(target.Bin.size()));
	//copy bin
	if (normalise) {
		//sum everything in the accumulator
		const float sum = static_cast<float>(std::reduce(acc_arr.first, acc_arr.second, 0u, [](auto init, const STPBin& bin) { return init + bin.Data.Quantity; }));
		std::transform(
			acc_arr.first, 
			acc_arr.second,
			std::back_inserter(target.Bin),
			[sum](STPBin bin) {
				//we need to make a copy
				bin.Data.Weight = 1.0f * bin.Data.Quantity / sum;
				return bin;
			}
		);
	}
	else {
		//just copy the data
		std::copy(acc_arr.first, acc_arr.second, std::back_inserter(target.Bin));
	}
}

void STPSingleHistogramFilter::filter_horizontal(const STPFreeSlipSampleManager& sample_map, unsigned int horizontal_start_offset, uvec2 h_range, unsigned char threadID, unsigned int radius) {
	STPHistogramBuffer& target = this->Cache[threadID];
	STPAccumulator& acc = this->Accumulator[threadID];
	//make sure both of them are cleared (don't deallocate)
	target.clear();
	acc.clear();

	//we assume the radius never goes out of the free-slip boundary
	for (unsigned int i = h_range.x; i < h_range.y; i++) {
		//the target (current) pixel index
		const unsigned int ti = i * sample_map.Data->FreeSlipRange.x + horizontal_start_offset;
			//the pixel index of left-most radius (inclusive of the current radius)
		unsigned int li = ti - radius,
			//the pixel index of right-most radius (exclusive of the current radius)
			ri = ti + radius + 1u;

		//load radius strip for the first pixel into accumulator
		//we exclude the right most pixel in the radius to avoid duplicate loading later.
		for (int j = -static_cast<int>(radius); j <= static_cast<int>(radius); j++) {
			acc.inc(sample_map[ti + j], 1u);
		}
		//copy the first pixel radius to buffer
		STPSingleHistogramFilter::copy_to_buffer(target, acc, false);
		//generate histogram, starting from the second pixel, we only loop through the central texture
		for (unsigned int j = 1u; j < sample_map.Data->Dimension.x; j++) {
			//load one pixel to the right while unloading one pixel from the left
			acc.inc(sample_map[ri++], 1u);
			acc.dec(sample_map[li++], 1u);
			
			//copy acc to buffer
			//advance to the next central pixel
			STPSingleHistogramFilter::copy_to_buffer(target, acc, false);
		}

		//clear the accumulator before starting the next row
		acc.clear();
	}
}

void STPSingleHistogramFilter::copy_to_output(unsigned char threadID, uvec2 output_base) {
	STPHistogramBuffer& target = this->Cache[threadID];
	auto offset_base_it = this->Output->HistogramStartOffset.begin() + output_base.y;
	//caller should guarantee the output container has been allocated that many elements, we are not doing back_inserter here

	//copy histogram offset
	if (threadID != 0u) {
		//do a offset correction first
		//no need to do that for thread 0 since offset starts at zero
		std::transform(
			target.HistogramStartOffset.cbegin(), 
			target.HistogramStartOffset.cend(), 
			offset_base_it,
			//get the starting index, so the current buffer connects to the previous buffer seamlessly
			[bin_base = output_base.x](auto offset) { return bin_base + offset; }
		);
	}
	else {
		//direct copy for threadID 0
		std::copy(target.HistogramStartOffset.cbegin(), target.HistogramStartOffset.cend(), offset_base_it);
	}

	//copy the bin
	std::copy(target.Bin.cbegin(), target.Bin.cend(), this->Output->Bin.begin() + output_base.x);
}

void STPSingleHistogramFilter::filter_vertical(const uvec2& dimension, uvec2 w_range, unsigned char threadID, unsigned int radius) {
	STPHistogramBuffer& target = this->Cache[threadID];
	STPAccumulator& acc = this->Accumulator[threadID];
	//clear both
	target.clear();
	acc.clear();

	//we use the output from horizontal pass as "texture", and assume the output pixels are always available
	for (unsigned int i = w_range.x; i < w_range.y; i++) {
		//the target (current) pixel index
		const unsigned int ti = i + radius * dimension.x;
			//the pixel index of up-most radius (inclusive of the current radius)
		unsigned int ui = i /* ti - radius * dimension.x */,
			//the pixel index of down-most radius (exclusive of the current radius)
			di = ti + (radius + 1u) * dimension.x;

		//load the radius into accumulator
		for (int j = -static_cast<int>(radius); j <= static_cast<int>(radius); j++) {
			auto bin_offset = this->Output->HistogramStartOffset.cbegin() + (ti + j * dimension.x);
			const unsigned int bin_start = *bin_offset,
				bin_end = *(bin_offset + 1);
			//it will be a bit tricky for the last pixel in the histogram since that's the last iterator in start offset array.
			//we have emplaced one more offset at the end of HistogramStartOffset in Output to indicate the size of the entire flatten Bin
			for (unsigned int bin_index = bin_start; bin_index < bin_end; bin_index++) {
				const STPBin& curr_bin = this->Output->Bin[bin_index];
				acc.inc(curr_bin.Item, curr_bin.Data.Quantity);
			}
		}
		//copy the first pixel to buffer
		//we can start normalising data on the go, the accumulator is complete for this pixel
		STPSingleHistogramFilter::copy_to_buffer(target, acc, true);
		//generate histogram
		for (unsigned int j = 1u; j < dimension.y; j++) {
			//load one pixel to the bottom while unloading one pixel from the top
			auto bin_beg = this->Output->HistogramStartOffset.cbegin();
			auto bin_offset_d = bin_beg + di,
				bin_offset_u = bin_beg + ui;
			//collect histogram at the bottom pixel
			{
				const unsigned int bin_start = *bin_offset_d,
					bin_end = *(bin_offset_d + 1);
				for (unsigned int bin_index = bin_start; bin_index < bin_end; bin_index++) {
					const STPBin& curr_bin = this->Output->Bin[bin_index];
					acc.inc(curr_bin.Item, curr_bin.Data.Quantity);
				}
			}
			//discard histogram at the top pixel
			{
				const unsigned int bin_start = *bin_offset_u,
					bin_end = *(bin_offset_u + 1);
				for (unsigned int bin_index = bin_start; bin_index < bin_end; bin_index++) {
					const STPBin& curr_bin = this->Output->Bin[bin_index];
					acc.dec(curr_bin.Item, curr_bin.Data.Quantity);
				}
			}

			//copy the accumulator to buffer
			STPSingleHistogramFilter::copy_to_buffer(target, acc, true);

			//advance to the next central pixel
			ui += dimension.x;
			di += dimension.x;
		}

		//clear the accumulator
		acc.clear();
	}
}

const STPSingleHistogramFilter::STPHistogramBuffer* STPSingleHistogramFilter::filter(const STPFreeSlipSampleManager& sample_map, const uvec2& central_chunk_index, unsigned int radius) {
	using namespace std::placeholders;
	using std::bind;
	using std::future;
	using std::ref;
	auto horizontal = bind(&STPSingleHistogramFilter::filter_horizontal, this, _1, _2, _3, _4, _5);
	auto vertical = bind(&STPSingleHistogramFilter::filter_vertical, this, _1, _2, _3, _4);
	auto copy_output = bind(&STPSingleHistogramFilter::copy_to_output, this, _1, _2);
	future<void> workgroup[STPSingleHistogramFilter::DEGREE_OF_PARALLELISM];
	//calculate central texture starting index
	const uvec2& dimension = reinterpret_cast<const uvec2&>(sample_map.Data->Dimension),
		central_starting_coordinate = dimension * central_chunk_index;

	auto sync_then_copy_to_output = [this, &copy_output, &workgroup]() -> void {
		size_t bin_total = 0ull,
			offset_total = 0ull;

		//sync working threads and get the total length of all buffers
		for (unsigned char w = 0u; w < STPSingleHistogramFilter::DEGREE_OF_PARALLELISM; w++) {
			workgroup[w].get();

			STPHistogramBuffer& curr_buffer = this->Cache[w];
			bin_total += curr_buffer.Bin.size();
			offset_total += curr_buffer.HistogramStartOffset.size();
		}

		//copy thread buffer to output
		//we don't need to clear the output, but rather we can resize it (items will get overwriten anyway)
		this->Output->Bin.resize(bin_total);
		this->Output->HistogramStartOffset.resize(offset_total);
		uvec2 base(0u);
		for (unsigned char w = 0u; w < STPSingleHistogramFilter::DEGREE_OF_PARALLELISM; w++) {
			workgroup[w] = this->filter_worker.enqueue_future(copy_output, w, base);

			//get the base index for the next worker, so each worker only copies buffer belongs to them to independent location
			STPHistogramBuffer& curr_buffer = this->Cache[w];
			//bin_base
			base.x += curr_buffer.Bin.size();
			//offset_base
			base.y += curr_buffer.HistogramStartOffset.size();
		}
		//sync
		for (unsigned char w = 0u; w < STPSingleHistogramFilter::DEGREE_OF_PARALLELISM; w++) {
			workgroup[w].get();
		}
		//in vertical pass, each pixel needs to access the current offset, and the offset of the next pixel.
		//insert one at the back to make sure the last pixel can read some valid data
		this->Output->HistogramStartOffset.emplace_back(static_cast<unsigned int>(this->Output->Bin.size()));
	};

	//perform horizontal filter
	//we need to start from the top halo, and ends at the bottom halo, width of which is radius.
	const unsigned int height_start = central_starting_coordinate.y - radius,
		//no need to minus 1 since our loop ends at less than (no equal)
		//we had already make sure radius is an even number to ensure divisibility, also it's not too large to go out of memory bound
		height_step = (dimension.y + 2u * radius) / STPSingleHistogramFilter::DEGREE_OF_PARALLELISM;
	uvec2 h_range(height_start, height_start + height_step);
	for (unsigned char w = 0u; w < STPSingleHistogramFilter::DEGREE_OF_PARALLELISM; w++) {
		workgroup[w] = this->filter_worker.enqueue_future(horizontal, ref(sample_map), central_starting_coordinate.x, h_range, w, radius);
		//increment range
		h_range.x = h_range.y;
		h_range.y += height_step;
	}
	//sync get the total length of all buffers and copy buffer to output
	sync_then_copy_to_output();

	//perform vertical filter
	//unlike horizontal pass, we start from the firts pixel of output from previous stage, and the output contains the free-slip texture.
	//width start from 0, output buffer has the same width as each texture, and 2 * radius addition to the height as halos
	const unsigned int width_step = dimension.x / STPSingleHistogramFilter::DEGREE_OF_PARALLELISM;
	uvec2 w_range(0u, width_step);
	for (unsigned char w = 0u; w < STPSingleHistogramFilter::DEGREE_OF_PARALLELISM; w++) {
		workgroup[w] = this->filter_worker.enqueue_future(vertical, ref(dimension), w_range, w, radius);
		//increment
		w_range.x = w_range.y;
		w_range.y += width_step;
	}
	//sync, do the same thing as what horizontal did
	sync_then_copy_to_output();

	//finally
	return this->Output.get();
}

STPSingleHistogramFilter::STPFilterReport STPSingleHistogramFilter::operator()(const STPFreeSlipSampleManager& sample_map, unsigned int radius) {
	//do some simple runtime check
	//before everything else, make sure user has told us the previously returned filter report can be destroyed
	if (this->ReportInUsed) {
		throw std::logic_error("a previously returned filter report is preventing execution, call destroy() first");
	}
	//first make sure radius is an even number
	if ((radius | 0x00u) == 0x00u) {
		throw std::invalid_argument("radius should be an even number");
	}
	//second make sure radius is not larger than the free-slip range
	//reinterpret_cast is safe as uint2 and uvec2 are both same in standard, alignment and type
	const uvec2 central_chunk_index = static_cast<uvec2>(glm::floor(vec2(reinterpret_cast<const uvec2&>(sample_map.Data->FreeSlipChunk)) / 2.0f)),
		halo_size = central_chunk_index * reinterpret_cast<const uvec2&>(sample_map.Data->Dimension);
	if (halo_size.x < radius || halo_size.y < radius) {
		throw std::invalid_argument("radius is too large and will overflow free-slip boundary");
	}

	//looks safe now, start the filter
	const STPHistogramBuffer* output_histogram = this->filter(sample_map, central_chunk_index, radius);
	//wrap it to the report
	STPFilterReport report = { output_histogram->Bin.data(), output_histogram->HistogramStartOffset.data() };

	this->ReportInUsed = true;

	return report;
}

void STPSingleHistogramFilter::destroy() const {
	this->ReportInUsed = false;
}