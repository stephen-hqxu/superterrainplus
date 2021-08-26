#pragma once
#include <SuperAlgorithm+/STPSingleHistogramFilter.h>

//Error
#include <SuperError+/STPDeviceErrorHandler.h>
//CUDA
#include <cuda_runtime.h>

//Container
#include <vector>
//Util
#include <limits>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <functional>
#include <type_traits>

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

/* Pinned memory allocator */

template<class T>
struct STPPinnedAllocator {
public:

	typedef T value_type;

	STPPinnedAllocator() = default;

	//Allocator should be copiable
	template<class U>
	STPPinnedAllocator(const STPPinnedAllocator<U>&) {
		//nothing to copy...
	}

	T* allocate(size_t size) const {
		T* mem;
		STPcudaCheckErr(cudaMallocHost(&mem, size));
		return mem;
	}

	void deallocate(T* mem, size_t) const {
		STPcudaCheckErr(cudaFreeHost(mem));
	}

	//Allocators must be equal after the copy, meaning they should share the same memory pool
	template<class U>
	constexpr bool operator==(const STPPinnedAllocator<U>&) {
		//we use CUDA API so memory will always comes from the OS
		return true;
	}

	template<class U>
	constexpr bool operator!=(const STPPinnedAllocator<U>&) {
		return false;
	}

};

/* Private object implementations */

template<bool Pinned>
struct STPSingleHistogramFilter::STPHistogramBuffer {
private:

	//Choose pinned allocator or default allocator
	template<class T>
	using StrategicAlloc = std::conditional_t<Pinned, typename std::allocator_traits<STPPinnedAllocator<T>>::template rebind_alloc<T>, std::allocator<T>>;

public:

	STPHistogramBuffer() = default;

	STPHistogramBuffer(const STPHistogramBuffer&) = delete;

	STPHistogramBuffer(STPHistogramBuffer&&) = delete;

	STPHistogramBuffer& operator=(const STPHistogramBuffer&) = delete;

	STPHistogramBuffer& operator=(STPHistogramBuffer&&) = delete;

	//All flatten bins in all histograms
	vector<STPSingleHistogram::STPBin, StrategicAlloc<STPSingleHistogram::STPBin>> Bin;
	//Get the bin starting index for a pixel in the flatten bin array
	vector<unsigned int, StrategicAlloc<unsigned int>> HistogramStartOffset;

	/**
	 * @brief Clear containers in histogram buffer.
	 * It doesn't gaurantee to free up memory allocated inside, acting as memory pool which can be reused.
	*/
	inline void clear() noexcept {
		this->Bin.clear();
		this->HistogramStartOffset.clear();
	}

};

//We don't need explicit instantiation for STPHistogramBuffer since it's a private struct and the main class implements have forced the compiler to instantiate
//We don't need to export instantiations either since the external environment won't call any function in this class (opaque type)

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
	inline STPSingleHistogram::STPBin& operator[](Sample sample) {
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
			STPSingleHistogram::STPBin& bin = this->Bin.emplace_back();
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
	vector<STPSingleHistogram::STPBin> Bin;

	//Array of bins in accumulator
	typedef vector<STPSingleHistogram::STPBin>::const_iterator STPBinIterator;
	typedef pair<STPBinIterator, STPBinIterator> STPBinArray;

	/**
	 * @brief Increment the sample bin by count
	 * @param sample The sample bin that will be operated on
	 * @param count The number to increment
	*/
	void inc(Sample sample, unsigned int count) {
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
		STPSingleHistogram::STPBin& bin = this->Bin[static_cast<unsigned int>(bin_index)];
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
	inline STPBinArray operator()() const noexcept {
		return make_pair(this->Bin.cbegin(), this->Bin.cend());
	}

	/**
	 * @brief Clear all content in accumulator, but leave reserved memory
	*/
	inline void clear() noexcept {
		this->Dictionary.clear();
		this->Bin.clear();
	}

};

/* Histogram Filter implementation */

STPSingleHistogramFilter::STPSingleHistogramFilter() : filter_worker(STPSingleHistogramFilter::DEGREE_OF_PARALLELISM), 
	Cache(make_unique<STPDefaultHistogramBuffer[]>(STPSingleHistogramFilter::DEGREE_OF_PARALLELISM)), 
	Accumulator(make_unique<STPAccumulator[]>(STPSingleHistogramFilter::DEGREE_OF_PARALLELISM)) {

}

STPSingleHistogramFilter::~STPSingleHistogramFilter() = default;

void STPSingleHistogramFilter::copy_to_buffer(STPDefaultHistogramBuffer& target, STPAccumulator& acc, bool normalise) {
	auto [acc_beg, acc_end] = acc();
	target.HistogramStartOffset.emplace_back(static_cast<unsigned int>(target.Bin.size()));

	//copy bin
	if (normalise) {
		//sum everything in the accumulator
		const float sum = static_cast<float>(std::reduce(acc_beg, acc_end, 0u, [](auto init, const STPSingleHistogram::STPBin& bin) { return init + bin.Data.Quantity; }));
		std::transform(
			acc_beg,
			acc_end,
			std::back_inserter(target.Bin),
			[sum](STPSingleHistogram::STPBin bin) {
				//we need to make a copy
				bin.Data.Weight = 1.0f * bin.Data.Quantity / sum;
				return bin;
			}
		);
	}
	else {
		//just copy the data
		std::copy(acc_beg, acc_end, std::back_inserter(target.Bin));
	}
}

void STPSingleHistogramFilter::filter_vertical(const STPFreeSlipSampleManager& sample_map, unsigned int vertical_start_offset, uvec2 w_range, unsigned char threadID, unsigned int radius) {
	STPDefaultHistogramBuffer& target = this->Cache[threadID];
	STPAccumulator& acc = this->Accumulator[threadID];
	//clear both
	target.clear();
	acc.clear();

	//we assume the radius never goes out of the free-slip boundary
	//we are traversing the a row-major sample_map column by column
	for (unsigned int i = w_range.x; i < w_range.y; i++) {
		//the target (current) pixel index
		const unsigned int ti = i + vertical_start_offset * sample_map.Data->FreeSlipRange.x;
		//the pixel index of up-most radius (inclusive of the current radius)
		unsigned int ui = ti - radius * sample_map.Data->FreeSlipRange.x,
			//the pixel index of down-most radius (exclusive of the current radius)
			di = ti + (radius + 1u) * sample_map.Data->FreeSlipRange.x;

		//load the radius into accumulator
		for (int j = -static_cast<int>(radius); j <= static_cast<int>(radius); j++) {
			acc.inc(sample_map[ti + j * sample_map.Data->FreeSlipRange.x], 1u);
		}
		//copy the first pixel to buffer
		STPSingleHistogramFilter::copy_to_buffer(target, acc, false);
		//generate histogram
		for (unsigned int j = 1u; j < sample_map.Data->Dimension.y; j++) {
			//load one pixel to the bottom while unloading one pixel from the top
			acc.inc(sample_map[di], 1u);
			acc.dec(sample_map[ui], 1u);

			//copy the accumulator to buffer
			STPSingleHistogramFilter::copy_to_buffer(target, acc, false);

			//advance to the next central pixel
			di += sample_map.Data->FreeSlipRange.x;
			ui += sample_map.Data->FreeSlipRange.x;
		}

		//clear the accumulator
		acc.clear();
	}
}

void STPSingleHistogramFilter::copy_to_output(STPPinnedHistogramBuffer* buffer, unsigned char threadID, uvec2 output_base) {
	STPDefaultHistogramBuffer& target = this->Cache[threadID];
	auto offset_base_it = buffer->HistogramStartOffset.begin() + output_base.y;
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
	std::copy(target.Bin.cbegin(), target.Bin.cend(), buffer->Bin.begin() + output_base.x);
}

void STPSingleHistogramFilter::filter_horizontal(STPPinnedHistogramBuffer* histogram_input, const uvec2& dimension, uvec2 h_range, unsigned char threadID, unsigned int radius) {
	STPDefaultHistogramBuffer& target = this->Cache[threadID];
	STPAccumulator& acc = this->Accumulator[threadID];
	//make sure both of them are cleared (don't deallocate)
	target.clear();
	acc.clear();

	//we use the output from vertical pass as "texture", and assume the output pixels are always available
	//remind that the Output from vertical stage is a column-major matrix and we are traversing it row by row
	for (unsigned int i = h_range.x; i < h_range.y; i++) {
		//the target (current) pixel index
		const unsigned int ti = i + radius * dimension.y;
		//the pixel index of left-most radius (inclusive of the current radius)
		unsigned int li = i /* ti - radius * dimension.y */,
			//the pixel index of right-most radius (exclusive of the current radius)
			ri = ti + (radius + 1u) * dimension.y;

		//load radius strip for the first pixel into accumulator
		for (int j = -static_cast<int>(radius); j <= static_cast<int>(radius); j++) {
			auto bin_offset = histogram_input->HistogramStartOffset.cbegin() + (ti + j * dimension.y);
			const unsigned int bin_start = *bin_offset,
				bin_end = *(bin_offset + 1);
			//it will be a bit tricky for the last pixel in the histogram since that's the last iterator in start offset array.
			//we have emplaced one more offset at the end of HistogramStartOffset in Output to indicate the size of the entire flatten Bin
			for (unsigned int bin_index = bin_start; bin_index < bin_end; bin_index++) {
				const STPSingleHistogram::STPBin& curr_bin = histogram_input->Bin[bin_index];
				acc.inc(curr_bin.Item, curr_bin.Data.Quantity);
			}
		}
		//copy the first pixel radius to buffer
		//we can start normalising data on the go, the accumulator is complete for this pixel
		STPSingleHistogramFilter::copy_to_buffer(target, acc, true);
		//generate histogram, starting from the second pixel, we only loop through the central texture
		for (unsigned int j = 1u; j < dimension.x; j++) {
			//load one pixel to the right while unloading one pixel from the left
			auto bin_beg = histogram_input->HistogramStartOffset.cbegin();
			auto bin_offset_r = bin_beg + ri,
				bin_offset_l = bin_beg + li;
			//collect histogram at the right pixel
			{
				const unsigned int bin_start = *bin_offset_r,
					bin_end = *(bin_offset_r + 1);
				for (unsigned int bin_index = bin_start; bin_index < bin_end; bin_index++) {
					const STPSingleHistogram::STPBin& curr_bin = histogram_input->Bin[bin_index];
					acc.inc(curr_bin.Item, curr_bin.Data.Quantity);
				}
			}
			//discard histogram at the left pixel
			{
				const unsigned int bin_start = *bin_offset_l,
					bin_end = *(bin_offset_l + 1);
				for (unsigned int bin_index = bin_start; bin_index < bin_end; bin_index++) {
					const STPSingleHistogram::STPBin& curr_bin = histogram_input->Bin[bin_index];
					acc.dec(curr_bin.Item, curr_bin.Data.Quantity);
				}
			}

			//copy acc to buffer
			STPSingleHistogramFilter::copy_to_buffer(target, acc, true);

			//advance to the next central pixel
			ri += dimension.y;
			li += dimension.y;
		}

		//clear the accumulator before starting the next row
		acc.clear();
	}
}

void STPSingleHistogramFilter::filter
	(const STPFreeSlipSampleManager& sample_map, STPPinnedHistogramBuffer* histogram_output, const uvec2& central_chunk_index, unsigned int radius) {
	using namespace std::placeholders;
	using std::bind;
	using std::future;
	using std::cref;
	auto vertical = bind(&STPSingleHistogramFilter::filter_vertical, this, _1, _2, _3, _4, _5);
	auto copy_output = bind(&STPSingleHistogramFilter::copy_to_output, this, _1, _2, _3);
	auto horizontal = bind(&STPSingleHistogramFilter::filter_horizontal, this, _1, _2, _3, _4, _5);
	future<void> workgroup[STPSingleHistogramFilter::DEGREE_OF_PARALLELISM];
	//calculate central texture starting index
	const uvec2& dimension = reinterpret_cast<const uvec2&>(sample_map.Data->Dimension),
		central_starting_coordinate = dimension * central_chunk_index;

	auto sync_then_copy_to_output = [this, histogram_output, &copy_output, &workgroup]() -> void {
		size_t bin_total = 0ull,
			offset_total = 0ull;

		//sync working threads and get the total length of all buffers
		for (unsigned char w = 0u; w < STPSingleHistogramFilter::DEGREE_OF_PARALLELISM; w++) {
			workgroup[w].get();

			STPDefaultHistogramBuffer& curr_buffer = this->Cache[w];
			bin_total += curr_buffer.Bin.size();
			offset_total += curr_buffer.HistogramStartOffset.size();
		}

		//copy thread buffer to output
		//we don't need to clear the output, but rather we can resize it (items will get overwriten anyway)
		histogram_output->Bin.resize(bin_total);
		histogram_output->HistogramStartOffset.resize(offset_total);
		uvec2 base(0u);
		for (unsigned char w = 0u; w < STPSingleHistogramFilter::DEGREE_OF_PARALLELISM; w++) {
			workgroup[w] = this->filter_worker.enqueue_future(copy_output, histogram_output, w, base);

			//get the base index for the next worker, so each worker only copies buffer belongs to them to independent location
			STPDefaultHistogramBuffer& curr_buffer = this->Cache[w];
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
		histogram_output->HistogramStartOffset.emplace_back(static_cast<unsigned int>(histogram_output->Bin.size()));
	};

	//we perform vertical first as the input sample_map is a row-major matrix, after this stage the output buffer will be in column-major.
	//then we perform horizontal filter on the output, so the final histogram will still be in row-major.
	//It seems like breaking the cache locality on sample_map but benchmark shows there's no difference in performance.

	//perform vertical filter
	{
		//we need to start from the left halo, and ends at the right halo, width of which is radius.
		//our loop in the filter ends at less than (no equal)
		//we had already make sure radius is an even number to ensure divisibility, also it's not too large to go out of memory bound
		const unsigned int width_start = central_starting_coordinate.x - radius,
			width_step = (dimension.x + 2u * radius) / STPSingleHistogramFilter::DEGREE_OF_PARALLELISM;
		uvec2 w_range(width_start, width_start + width_step);
		for (unsigned char w = 0u; w < STPSingleHistogramFilter::DEGREE_OF_PARALLELISM; w++) {
			workgroup[w] = this->filter_worker.enqueue_future(vertical, cref(sample_map), central_starting_coordinate.y, w_range, w, radius);
			//increment
			w_range.x = w_range.y;
			w_range.y += width_step;
		}
		//sync get the total length of all buffers and copy buffer to output
		sync_then_copy_to_output();
	}
	//perform horizontal filter
	{
		//unlike vertical pass, we start from the firts pixel of output from previous stage, and the output contains the halo histogram.
		//height start from 0, output buffer has the same height as each texture, and 2 * radius addition to the horizontal width as halos
		const unsigned int height_step = dimension.y / STPSingleHistogramFilter::DEGREE_OF_PARALLELISM;
		uvec2 h_range(0u, height_step);
		for (unsigned char w = 0u; w < STPSingleHistogramFilter::DEGREE_OF_PARALLELISM; w++) {
			workgroup[w] = this->filter_worker.enqueue_future(horizontal, histogram_output, cref(dimension), h_range, w, radius);
			//increment range
			h_range.x = h_range.y;
			h_range.y += height_step;
		}
		//sync, do the same thing as what vertical did
		sync_then_copy_to_output();
	}

	//finished
}

STPSingleHistogramFilter::STPHistogramBuffer_t STPSingleHistogramFilter::createHistogramBuffer() {
	//I hate to use `new` but unfortunately make_unique doesn't work with custom deleter...
	return STPHistogramBuffer_t(new STPPinnedHistogramBuffer(), [](STPPinnedHistogramBuffer* buffer) { std::default_delete<STPPinnedHistogramBuffer>()(buffer); });
}

STPSingleHistogram STPSingleHistogramFilter::operator()(const STPFreeSlipSampleManager& samplemap_manager, const STPHistogramBuffer_t& histogram_output, unsigned int radius) {
	//do some simple runtime check
	//first make sure radius is an even number
	if ((radius | 0x00u) == 0x00u) {
		throw std::invalid_argument("radius should be an even number");
	}
	//second make sure radius is not larger than the free-slip range
	//reinterpret_cast is safe as uint2 and uvec2 are both same in standard, alignment and type
	const uvec2 central_chunk_index = static_cast<uvec2>(glm::floor(vec2(reinterpret_cast<const uvec2&>(samplemap_manager.Data->FreeSlipChunk)) / 2.0f));
	if (const uvec2 halo_size = central_chunk_index * reinterpret_cast<const uvec2&>(samplemap_manager.Data->Dimension); 
		halo_size.x < radius || halo_size.y < radius) {
		throw std::invalid_argument("radius is too large and will overflow free-slip boundary");
	}

	//looks safe now, start the filter
	this->filter(samplemap_manager, histogram_output.get(), central_chunk_index, radius);

	return STPSingleHistogramFilter::readHistogramBuffer(histogram_output);
}

STPSingleHistogram STPSingleHistogramFilter::readHistogramBuffer(const STPHistogramBuffer_t& buffer) {
	return STPSingleHistogram{ buffer->Bin.data(), buffer->HistogramStartOffset.data() };
}