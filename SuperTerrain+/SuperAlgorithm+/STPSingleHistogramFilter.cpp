#include <SuperAlgorithm+/STPSingleHistogramFilter.h>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
//CUDA
#include <cuda_runtime.h>

//Utility
#include <limits>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <functional>
#include <type_traits>

#include <stdexcept>
#include <cstring>

using namespace SuperTerrainPlus::STPCompute;
using SuperTerrainPlus::STPDiversity::Sample;

using glm::vec2;
using glm::uvec2;

using std::unique_ptr;
using std::make_unique;
using std::pair;
using std::make_pair;
using std::unique_lock;
using std::mutex;
using std::allocator;
using std::allocator_traits;

/* Pinned memory allocator */

template<class T>
struct STPPinnedAllocator {
public:

	typedef T value_type;

	T* allocate(size_t size) const {
		T* mem;
		//note that the allocator-provided size is the number of object, not byte
		STPcudaCheckErr(cudaMallocHost(&mem, size * sizeof(T)));
		return mem;
	}

	void deallocate(T* mem, size_t) const {
		STPcudaCheckErr(cudaFreeHost(mem));
	}

};

/* Custom data structure */

/**
 * @brief A simple implementation of std::vector.
 * After some profiling a custom data structure works better than the stl one due to its simplicity.
 * @tparam T The data type of the array list
*/
template<class T, class A = allocator<T>>
class STPArrayList {
public:

	//The iterator for the array list
	using STPArrayList_it = T*;
	//The const iterator for the array list
	using STPArrayList_cit = const T*;

private:

	static_assert(std::is_trivial_v<T>, "type must be a trivial type");
	static_assert(std::is_trivially_destructible_v<T>, "type must be trivially destructible");

	using RebindAlloc = typename allocator_traits<A>::template rebind_alloc<T>;
	using AllocTr = allocator_traits<RebindAlloc>;

	RebindAlloc arrayAllocator;

	//Smart pointer that uses allocator to destroy
	using unique_ptr_alloc = unique_ptr<T[], std::function<void(T*)>>;

	//The data held by the array list, also denotes the beginning of the array
	unique_ptr_alloc Begin;
	//The pointer to the end of the array.
	STPArrayList_it End;
	//The pointer to the end of the internal storage array
	STPArrayList_it Last;

	/**
	 * @brief Expand the internal storage capacity
	 * @param new_capacity The new capacity
	*/
	void expand(size_t new_capacity) {
		using std::bind;
		using namespace std::placeholders;
		//clamp the capacity, because a newly created array list might have zero capacity.
		new_capacity = std::max(static_cast<size_t>(1u), new_capacity);
		//the current size of the user array
		const size_t current_size = this->size();

		//allocate a cache
		constexpr static auto deleter = [](T* ptr, RebindAlloc alloc, size_t size) constexpr -> void {
			//ptr is trivially destructor so we don't need to call destroy
			AllocTr::deallocate(alloc, ptr, size);
		};
		unique_ptr_alloc cache(AllocTr::allocate(this->arrayAllocator, new_capacity), bind(deleter, _1, this->arrayAllocator, new_capacity));

		//copy will handle the case when begin == end
		std::copy(this->cbegin(), this->cend(), cache.get());
		
		//reassign data
		this->Begin = std::move(cache);
		this->End = this->begin() + current_size;
		this->Last = this->begin() + new_capacity;
	}

	/**
	 * @brief Check if reallocation is needed for insert_back_n function
	 * @param count The number of new elements are going to be inserted
	*/
	inline void insert_back_realloc_check(size_t count) {
		if (this->cend() + count > this->Last) {
			//not enough room, reallocation
			this->expand((this->size() + count) * 2ull);
		}
	}

	/**
	 * @brief Insert count many elements at the end without initialisation
	 * @param count The number of element to be inserted
	*/
	inline void insert_back_n(size_t count) {
		this->insert_back_realloc_check(count);

		//simply move the end iterator ahead
		this->End += count;
	}

public:

	constexpr STPArrayList() noexcept : End(nullptr), Last(nullptr) {

	}

	STPArrayList(const STPArrayList&) = delete;

	STPArrayList(STPArrayList&&) = delete;

	STPArrayList& operator=(const STPArrayList&) = delete;

	STPArrayList& operator=(STPArrayList&&) = delete;

	~STPArrayList() = default;

	/**
	 * @brief Construct element in-place and put the newly constructed element at the end of the array list
	 * @tparam ...Arg Argument list
	 * @param ...arg All arguments to be used to construct the element
	 * @return The reference to the new element
	*/
	template<class... Arg>
	T& emplace_back(Arg&&... arg) {
		//check if we have free capacity
		if (this->End == this->Last) {
			//no more free room, expand, capacity is clamped in the function
			this->expand(this->capacity() * 2ull);
		}

		T& item = *this->end();
		//this is actually pretty dangerous.
		//if the type has non-trivial destructor and it will be called on the garbage data at the end iterator, which results in undefined behaviour.
		//but it's fine since we are dealing with trivial type only
		item = T(std::forward<Arg>(arg)...);
		
		//finally push the end by 1
		this->End++;
		return item;
	}

	/**
	 * @brief Insert count many elements at the end with initialisation
	 * @param count The number of element to be inserted
	 * @param value The value to be filled for every new element
	 * @return The iterator to the last inserted element
	*/
	STPArrayList_it insert_back_n(size_t count, const T& value) {
		this->insert_back_realloc_check(count);

		//init
		std::fill(this->end(), this->end() + count, value);

		this->End += count;

		return this->end() - 1;
	}

	/**
	 * @brief Erase the item pointed by the iterator
	 * Erasure of an empty container is undefined behaviour
	 * @param it The item to be erased
	 * @return The iterator to the item following the iterator provided
	*/
	STPArrayList_it erase(STPArrayList_it it) {
		//we don't need to call the destructor since T is always trivially destructible

		if (it < this->cend() - 1ull) {
			//it's not the last element, we need to move the memory forward
			//there's nothing to move if it's the last element, plus it may trigger undefined behaviour because end pointer should not be dereferenced
			//move_start is out of the range of [first, last) so we are good to use
			//no need to use std::move because T is known to be a PoD
			std::copy(const_cast<STPArrayList_cit>(it) + 1ull, this->cend(), it);
		}
		
		//correct the internal iterator
		this->End--;
		return it;
	}

	/**
	 * @brief Resize the array, either allocating more space or shrink it.
	 * Since T is trivial, no initialisation will be done to save time.
	 * @param new_size The new size for the array
	*/
	void resize(size_t new_size) {
		const size_t current_size = this->size();

		if (new_size == current_size) {
			//nothing needs to be done
			return;
		}
		if (new_size < current_size) {
			//simply move the end iterator by that many number
			this->End = this->begin() + new_size;
			return;
		}
		if (new_size > current_size) {
			//tricky, either just move the iterator ahead, or reallocation
			this->insert_back_n(new_size - current_size);
		}
	}

	/**
	 * @brief Get the reference to the element in the index
	 * @param index The index of the element to be retrieved
	 * @return The reference.
	 * If index is out-of-bound, return value is undefined.
	*/
	inline T& operator[](size_t index) {
		return this->Begin[index];
	}

	/**
	 * @brief Clear the internal storage of the array list.
	 * Clear takes constant time as the array list only deals with trivial type
	*/
	inline void clear() {
		//since the array list only holds trivially destructible type, we can simply move the pointer
		this->End = this->begin();
	}

	/**
	 * @brief Check if the array list is empty.
	 * @return True if the array list is empty.
	 * True denotes size equals zero
	*/
	inline bool empty() const {
		return this->size() == 0ull;
	}

	/**
	 * @brief Retrieve the number of element stored in the array list
	 * @return The number of element
	*/
	inline size_t size() const {
		return this->End - this->Begin.get();
	}

	/**
	 * @brief Get the number of element can be held by the internal storage
	 * @return The capacity
	*/
	inline size_t capacity() const {
		return this->Last - this->Begin.get();
	}

	/**
	 * @brief Get the constant iterator to the beginning of the array list
	 * @return The const iterator to the beginning of the array list
	*/
	inline STPArrayList_cit cbegin() const {
		return this->Begin.get();
	}

	/**
	 * @brief Get the constant iterator to the end of the array list
	 * @return The const iterator to the end of the array list
	*/
	inline STPArrayList_cit cend() const {
		return this->End;
	}

	/**
	 * @brief Get the iterator to the beginning of the array list
	 * @return The iterator to the beginning of the array list
	*/
	inline STPArrayList_it begin() {
		return const_cast<STPArrayList_it>(const_cast<const STPArrayList<T, A>*>(this)->cbegin());
	}

	/**
	 * @brief Get the iterator to the end of the array list
	 * @return The iterator to the end of the array list.
	 * Note that dereferencing this iterator will result in undefined behaviour.
	*/
	inline STPArrayList_it end() {
		return const_cast<STPArrayList_it>(const_cast<const STPArrayList<T, A>*>(this)->cend());
	}

	/**
	 * @brief Get the pointer to the first element in the array list.
	 * @return The pointer to the first element.
	 * If array list is empty, return nullptr
	*/
	inline T* data() {
		return this->begin();
	}

};

/* Private object implementations */

template<bool Pinned>
struct STPSingleHistogramFilter::STPHistogramBuffer {
private:

	//Choose pinned allocator or default allocator
	template<class T>
	using StrategicAlloc = typename std::conditional_t<Pinned, STPPinnedAllocator<T>, std::allocator<T>>;

public:

	STPHistogramBuffer() noexcept = default;

	STPHistogramBuffer(const STPHistogramBuffer&) = delete;

	STPHistogramBuffer(STPHistogramBuffer&&) = delete;

	STPHistogramBuffer& operator=(const STPHistogramBuffer&) = delete;

	STPHistogramBuffer& operator=(STPHistogramBuffer&&) = delete;

	~STPHistogramBuffer() = default;

	//All flatten bins in all histograms
	STPArrayList<STPSingleHistogram::STPBin, StrategicAlloc<STPSingleHistogram::STPBin>> Bin;
	//Get the bin starting index for a pixel in the flatten bin array
	STPArrayList<unsigned int, StrategicAlloc<unsigned int>> HistogramStartOffset;

	/**
	 * @brief Clear containers in histogram buffer.
	 * It doesn't guarantee to free up memory allocated inside, acting as memory pool which can be reused.
	*/
	inline void clear() noexcept {
		this->Bin.clear();
		this->HistogramStartOffset.clear();
	}

};

//We don't need explicit instantiation for STPHistogramBuffer since it's a private structure and the main class implements have forced the compiler to instantiate
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

		//get the biome using the index from dictionary
		//if not we need to insert that many extra entries so we can use sample to index the dictionary directly
		if (unsigned int& bin_index = diff <= 0 ? *this->Dictionary.insert_back_n(static_cast<size_t>((-diff) + 1u), NO_ENTRY) : this->Dictionary[sample]; 
			bin_index == NO_ENTRY) {
			//biome not exist, add and initialise
			STPSingleHistogram::STPBin& bin = this->Bin.emplace_back();
			bin.Item = sample;
			bin.Data.Quantity = 0u;
			//record the index in the bin and store to dictionary
			bin_index = static_cast<unsigned int>(this->Bin.size()) - 1;
			return bin;
		}
		else {
			return this->Bin[bin_index];
		}
	}

public:

	STPAccumulator() noexcept = default;

	STPAccumulator(const STPAccumulator&) = delete;

	STPAccumulator(STPAccumulator&&) = delete;

	STPAccumulator& operator=(const STPAccumulator&) = delete;

	STPAccumulator& operator=(STPAccumulator&&) = delete;

	~STPAccumulator() = default;

	//Use Sample as index, find the index in Bin for this sample
	STPArrayList<unsigned int> Dictionary;
	//Store the number of element
	STPArrayList<STPSingleHistogram::STPBin> Bin;

	//Array of bins in accumulator
	typedef STPArrayList<STPSingleHistogram::STPBin>::STPArrayList_cit STPBinIterator;
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
	void dec(Sample sample, unsigned int count) {
		//our algorithm guarantees the bin has been increment by this sample before, so no check is needed
		unsigned int& bin_index = this->Dictionary[sample];
		unsigned int& quant = this->Bin[static_cast<unsigned int>(bin_index)].Data.Quantity;

		if (quant <= count) {
			//update the dictionary entries linearly, basically it's a rehash in hash table
			auto followed_index = this->Bin.erase(this->Bin.begin() + bin_index);
			std::for_each(followed_index, this->Bin.end(), [&Dic = this->Dictionary](const auto& bin) {
				//since all subsequent indices followed by the erased bin has been advanced forward by one block
				//we need to subtract the indices recorded in dictionary for those entries by one.
				Dic[bin.Item]--;
			});

			//bin will become empty, erase this bin and dictionary entry
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

void STPSingleHistogramFilter::STPPinnedHistogramBufferDeleter::operator()(STPPinnedHistogramBuffer* ptr) const {
	std::default_delete<STPPinnedHistogramBuffer>()(ptr);
}

STPSingleHistogramFilter::STPSingleHistogramFilter() : filter_worker(STPSingleHistogramFilter::Parallelism) {

}

STPSingleHistogramFilter::~STPSingleHistogramFilter() = default;

STPSingleHistogramFilter::STPMemoryBlock STPSingleHistogramFilter::requestWorkplace() {
	unique_lock<mutex> write(this->CacheLock);

	//check if there is any free organisation
	if (this->MemoryBlockCache.empty()) {
		//request new
		return make_unique<STPWorkplace[]>(STPSingleHistogramFilter::Parallelism);
	}
	//get existing from the queue
	STPMemoryBlock block = std::move(this->MemoryBlockCache.front());
	this->MemoryBlockCache.pop();
	return block;
}

void STPSingleHistogramFilter::returnWorkplace(STPMemoryBlock& block) {
	unique_lock<mutex> write(this->CacheLock);

	//push the id back to the free working memory
	this->MemoryBlockCache.emplace(std::move(block));
}

void STPSingleHistogramFilter::copy_to_buffer(STPDefaultHistogramBuffer& target, STPAccumulator& acc, bool normalise) {
	const auto [acc_beg, acc_end] = acc();
	target.HistogramStartOffset.emplace_back(static_cast<unsigned int>(target.Bin.size()));

	//instead of using the slow back_inserter, we can resize the array first
	const size_t bin_size = acc_end - acc_beg;
	target.Bin.resize(target.Bin.size() + bin_size);
	auto target_dest_begin = target.Bin.end() - bin_size;
	//copy bin
	if (normalise) {
		//sum everything in the accumulator
		const float sum = static_cast<float>(std::accumulate(acc_beg, acc_end, 0u, [](auto init, const STPSingleHistogram::STPBin& bin) { return init + bin.Data.Quantity; }));
		std::transform(
			acc_beg,
			acc_end,
			target_dest_begin,
			[sum](auto bin) {
				//we need to make a copy
				bin.Data.Weight = 1.0f * bin.Data.Quantity / sum;
				return bin;
			}
		);
	}
	else {
		//just copy the data
		std::copy(acc_beg, acc_end, target_dest_begin);
	}
}

void STPSingleHistogramFilter::filter_vertical(const Sample* sample_map, unsigned int freeslip_rangeX, unsigned int dimensionY, 
	unsigned int vertical_start_offset, uvec2 w_range,
	STPWorkplace& workplace, unsigned int radius) {
	auto& [target, acc] = workplace;
	//clear both
	target.clear();
	acc.clear();

	//we assume the radius never goes out of the free-slip boundary
	//we are traversing the a row-major sample_map column by column
	for (unsigned int i = w_range.x; i < w_range.y; i++) {
		//the target (current) pixel index
		const unsigned int ti = i + vertical_start_offset * freeslip_rangeX;
		//the pixel index of up-most radius (inclusive of the current radius)
		unsigned int ui = ti - radius * freeslip_rangeX,
			//the pixel index of down-most radius (exclusive of the current radius)
			di = ti + (radius + 1u) * freeslip_rangeX;

		//load the radius into accumulator
		for (int j = -static_cast<int>(radius); j <= static_cast<int>(radius); j++) {
			acc.inc(sample_map[ti + j * freeslip_rangeX], 1u);
		}
		//copy the first pixel to buffer
		STPSingleHistogramFilter::copy_to_buffer(target, acc, false);
		//generate histogram
		for (unsigned int j = 1u; j < dimensionY; j++) {
			//load one pixel to the bottom while unloading one pixel from the top
			acc.inc(sample_map[di], 1u);
			acc.dec(sample_map[ui], 1u);

			//copy the accumulator to buffer
			STPSingleHistogramFilter::copy_to_buffer(target, acc, false);

			//advance to the next central pixel
			di += freeslip_rangeX;
			ui += freeslip_rangeX;
		}

		//clear the accumulator
		acc.clear();
	}
}

void STPSingleHistogramFilter::copy_to_output(STPPinnedHistogramBuffer* buffer, 
	const STPDefaultHistogramBuffer& workplace_memory, unsigned char workplaceID, uvec2 output_base) {
	auto offset_base_it = buffer->HistogramStartOffset.begin() + output_base.y;
	//caller should guarantee the output container has been allocated that many elements, we don't need to allocate memory here

	//copy histogram offset
	if (workplaceID != 0u) {
		//do a offset correction first
		//no need to do that for thread 0 since offset starts at zero
		std::transform(
			workplace_memory.HistogramStartOffset.cbegin(),
			workplace_memory.HistogramStartOffset.cend(),
			offset_base_it,
			//get the starting index, so the current buffer connects to the previous buffer seamlessly
			[bin_base = output_base.x](auto offset) { return bin_base + offset; }
		);
	}
	else {
		//direct copy for threadID 0
		std::copy(workplace_memory.HistogramStartOffset.cbegin(), workplace_memory.HistogramStartOffset.cend(), offset_base_it);
	}

	//copy the bin
	std::copy(workplace_memory.Bin.cbegin(), workplace_memory.Bin.cend(), buffer->Bin.begin() + output_base.x);
}

void STPSingleHistogramFilter::filter_horizontal(STPPinnedHistogramBuffer* histogram_input, const uvec2& dimension, uvec2 h_range, 
	STPWorkplace& workplace, unsigned int radius) {
	auto& [target, acc] = workplace;
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

			//copy accumulator to buffer
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
	(const Sample* sample_map, const STPFreeSlipInformation& freeslip_info, STPPinnedHistogramBuffer* histogram_output, uvec2 central_chunk_index, unsigned int radius) {
	using namespace std::placeholders;
	using std::bind;
	using std::future;
	using std::cref;
	using std::ref;

	auto vertical = bind(&STPSingleHistogramFilter::filter_vertical, this, _1, _2, _3, _4, _5, _6, _7);
	auto copy_output = bind(&STPSingleHistogramFilter::copy_to_output, this, _1, _2, _3, _4);
	auto horizontal = bind(&STPSingleHistogramFilter::filter_horizontal, this, _1, _2, _3, _4, _5);
	future<void> workgroup[STPSingleHistogramFilter::Parallelism];
	//calculate central texture starting index
	const uvec2& dimension = freeslip_info.Dimension,
		central_starting_coordinate = dimension * central_chunk_index;

	//request a working memory
	STPMemoryBlock memoryBlock = this->requestWorkplace();

	auto sync_then_copy_to_output = [this, histogram_output, &copy_output, &workgroup, &memoryBlock]() -> void {
		size_t bin_total = 0ull,
			offset_total = 0ull;

		//sync working threads and get the total length of all buffers
		for (unsigned char w = 0u; w < STPSingleHistogramFilter::Parallelism; w++) {
			workgroup[w].get();

			//grab the buffer in the allocated workplace belongs to this thread
			STPDefaultHistogramBuffer& curr_buffer = memoryBlock[w].first;
			bin_total += curr_buffer.Bin.size();
			offset_total += curr_buffer.HistogramStartOffset.size();
		}

		//copy thread buffer to output
		//we don't need to clear the output, but rather we can resize it (items will get overwritten anyway)
		histogram_output->Bin.resize(bin_total);
		histogram_output->HistogramStartOffset.resize(offset_total);
		uvec2 base(0u);
		for (unsigned char w = 0u; w < STPSingleHistogramFilter::Parallelism; w++) {
			workgroup[w] = this->filter_worker.enqueue_future(copy_output, histogram_output, cref(memoryBlock[w].first), w, base);

			//get the base index for the next worker, so each worker only copies buffer belongs to them to independent location
			STPDefaultHistogramBuffer& curr_buffer = memoryBlock[w].first;
			//bin_base
			base.x += static_cast<unsigned int>(curr_buffer.Bin.size());
			//offset_base
			base.y += static_cast<unsigned int>(curr_buffer.HistogramStartOffset.size());
		}
		//sync
		for (unsigned char w = 0u; w < STPSingleHistogramFilter::Parallelism; w++) {
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
			total_width = dimension.x + 2u * radius,
			width_step = total_width / STPSingleHistogramFilter::Parallelism;
		uvec2 w_range(width_start, width_start + width_step);
		for (unsigned char w = 0u; w < STPSingleHistogramFilter::Parallelism; w++) {
			workgroup[w] = this->filter_worker.enqueue_future(vertical, cref(sample_map), freeslip_info.FreeSlipRange.x, freeslip_info.Dimension.y, 
				central_starting_coordinate.y, w_range, ref(memoryBlock[w]), radius);
			//increment
			w_range.x = w_range.y;
			if (w == STPSingleHistogramFilter::Parallelism - 2u) {
				//calculate the range for the last thread, to ensure all remaining columns are all done by the last thread
				w_range.y = width_start + total_width;
			}
			else {
				w_range.y += width_step;
			}
		}
		//sync get the total length of all buffers and copy buffer to output
		sync_then_copy_to_output();
	}
	//perform horizontal filter
	{
		//unlike vertical pass, we start from the first pixel of output from previous stage, and the output contains the halo histogram.
		//height start from 0, output buffer has the same height as each texture, and 2 * radius addition to the horizontal width as halos
		const unsigned int height_step = dimension.y / STPSingleHistogramFilter::Parallelism;
		uvec2 h_range(0u, height_step);
		for (unsigned char w = 0u; w < STPSingleHistogramFilter::Parallelism; w++) {
			workgroup[w] = this->filter_worker.enqueue_future(horizontal, histogram_output, cref(dimension), h_range, ref(memoryBlock[w]), radius);
			//increment range
			h_range.x = h_range.y;
			if (w == STPSingleHistogramFilter::Parallelism - 2u) {
				h_range.y = dimension.y;
			}
			else {
				h_range.y += height_step;
			}
		}
		//sync, do the same thing as what vertical did
		sync_then_copy_to_output();
	}

	//finished, return working memory
	this->returnWorkplace(memoryBlock);
}

STPSingleHistogramFilter::STPHistogramBuffer_t STPSingleHistogramFilter::createHistogramBuffer() {
	//I hate to use `new` but unfortunately make_unique doesn't work with custom deleter...
	return STPHistogramBuffer_t(new STPPinnedHistogramBuffer());
}

STPSingleHistogram STPSingleHistogramFilter::operator()(const Sample* samplemap, const STPFreeSlipInformation& freeslip_info, 
	const STPHistogramBuffer_t& histogram_output, unsigned int radius) {
	//do some simple runtime check
	//first make sure radius is an even number
	if (radius == 0u || (radius & 0x01u) != 0x00u) {
		throw STPException::STPBadNumericRange("radius should be an positive even number");
	}
	//second make sure radius is not larger than the free-slip range
	const uvec2 central_chunk_index = freeslip_info.FreeSlipChunk / 2u;
	if (const uvec2 halo_size = central_chunk_index * freeslip_info.Dimension;
		halo_size.x < radius || halo_size.y < radius) {
		throw STPException::STPBadNumericRange("radius is too large and will overflow free-slip boundary");
	}

	//looks safe now, start the filter
	this->filter(samplemap, freeslip_info, histogram_output.get(), central_chunk_index, radius);

	return STPSingleHistogramFilter::readHistogramBuffer(histogram_output);
}

STPSingleHistogram STPSingleHistogramFilter::readHistogramBuffer(const STPHistogramBuffer_t& buffer) {
	return STPSingleHistogram{ buffer->Bin.data(), buffer->HistogramStartOffset.data() };
}