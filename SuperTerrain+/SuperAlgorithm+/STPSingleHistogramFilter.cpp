#include <SuperAlgorithm+/STPSingleHistogramFilter.h>

//CUDA
#include <cuda_runtime.h>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
//Threading
#include <SuperTerrain+/Utility/STPThreadPool.h>
//Memory
#include <SuperTerrain+/Utility/Memory/STPObjectPool.h>


//GLM
#include <glm/vec2.hpp>

//Utility
#include <limits>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <type_traits>

//Container
#include <utility>

#include <cstring>

using namespace SuperTerrainPlus::STPAlgorithm;
using SuperTerrainPlus::STPDiversity::Sample;

using glm::vec2;
using glm::uvec2;

using std::allocator;
using std::allocator_traits;
using std::pair;
using std::unique_ptr;

using std::make_pair;
using std::make_unique;

/* Pinned memory allocator */

template<class T>
struct STPPinnedAllocator {
public:

	typedef T value_type;

	T* allocate(const size_t size) const {
		T* mem;
		//note that the allocator-provided size is the number of object, not byte
		STP_CHECK_CUDA(cudaMallocHost(&mem, size * sizeof(T)));
		return mem;
	}

	void deallocate(T* const mem, const size_t) const {
		STP_CHECK_CUDA(cudaFreeHost(mem));
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
		constexpr static auto deleter = [](T* const ptr, RebindAlloc alloc, const size_t size) constexpr->void {
			//ptr is trivially destructor so we don't need to call destroy
			AllocTr::deallocate(alloc, ptr, size);
		};
		unique_ptr_alloc cache(AllocTr::allocate(this->arrayAllocator, new_capacity),
			bind(deleter, _1, this->arrayAllocator, new_capacity));

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
	inline void insert_back_realloc_check(const size_t count) {
		if (this->cend() + count > this->Last) {
			//not enough room, reallocation
			this->expand((this->size() + count) * 2u);
		}
	}

	/**
	 * @brief Insert count many elements at the end without initialisation
	 * @param count The number of element to be inserted
	*/
	inline void insert_back_n(const size_t count) {
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
			this->expand(this->capacity() * 2u);
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
	STPArrayList_it insert_back_n(const size_t count, const T& value) {
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
	STPArrayList_it erase(const STPArrayList_it it) {
		//we don't need to call the destructor since T is always trivially destructible

		if (it < this->cend() - 1) {
			//it's not the last element, we need to move the memory forward
			//there's nothing to move if it's the last element, plus it may trigger undefined behaviour because end pointer should not be dereferenced
			//move_start is out of the range of [first, last) so we are good to use
			//no need to use std::move because T is known to be a PoD
			std::copy(const_cast<STPArrayList_cit>(it) + 1, this->cend(), it);
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
	void resize(const size_t new_size) {
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
	inline T& operator[](const size_t index) noexcept {
		return this->Begin[index];
	}

	/**
	 * @brief Clear the internal storage of the array list.
	 * Clear takes constant time as the array list only deals with trivial type
	*/
	inline void clear() noexcept {
		//since the array list only holds trivially destructible type, we can simply move the pointer
		this->End = this->begin();
	}

	/**
	 * @brief Check if the array list is empty.
	 * @return True if the array list is empty.
	 * True denotes size equals zero
	*/
	inline bool empty() const noexcept {
		return this->size() == 0u;
	}

	/**
	 * @brief Retrieve the number of element stored in the array list
	 * @return The number of element
	*/
	inline size_t size() const noexcept {
		return this->End - this->Begin.get();
	}

	/**
	 * @brief Get the number of element can be held by the internal storage
	 * @return The capacity
	*/
	inline size_t capacity() const noexcept {
		return this->Last - this->Begin.get();
	}

	/**
	 * @brief Get the constant iterator to the beginning of the array list
	 * @return The const iterator to the beginning of the array list
	*/
	inline STPArrayList_cit cbegin() const noexcept {
		return this->Begin.get();
	}

	/**
	 * @brief Get the constant iterator to the end of the array list
	 * @return The const iterator to the end of the array list
	*/
	inline STPArrayList_cit cend() const noexcept {
		return this->End;
	}

	/**
	 * @brief Get the iterator to the beginning of the array list
	 * @return The iterator to the beginning of the array list
	*/
	inline STPArrayList_it begin() noexcept {
		return const_cast<STPArrayList_it>(const_cast<const STPArrayList<T, A>*>(this)->cbegin());
	}

	/**
	 * @brief Get the iterator to the end of the array list
	 * @return The iterator to the end of the array list.
	 * Note that dereferencing this iterator will result in undefined behaviour.
	*/
	inline STPArrayList_it end() noexcept {
		return const_cast<STPArrayList_it>(const_cast<const STPArrayList<T, A>*>(this)->cend());
	}

	/**
	 * @brief Get the pointer to the first element in the array list.
	 * @return The pointer to the first element.
	 * If array list is empty, return nullptr
	*/
	inline T* data() noexcept {
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

/* Single Histogram Filter implementation */

class STPSingleHistogramFilter::STPSHFKernel {
private:

	/**
	 * @brief Accumulator acts as a cache for each row or column iteration.
	 * The accumulator for the next pixel equals to the accumulator from the previous pixel plus the left/down
	 * out-of-range radius pixel and minus the right/up in-range radius pixel.
	*/
	class STPAccumulator {
	public:

		//Use Sample as index, find the index in Bin for this sample
		STPArrayList<unsigned int> Dictionary;
		//Store the number of element
		STPArrayList<STPSingleHistogram::STPBin> Bin;

		//Array of bins in accumulator
		typedef STPArrayList<STPSingleHistogram::STPBin>::STPArrayList_cit STPBinIterator;
		typedef pair<STPBinIterator, STPBinIterator> STPBinArray;

	private:

		//Denotes there is no entry in the dictionary that corresponds to the index in bin
		constexpr static unsigned int NO_ENTRY = std::numeric_limits<unsigned int>::max();

		/**
		 * @brief Get the bin for a biome sample.
		 * If bin doesn't exist, a new bin is created.
		 * @param sample The biome sample
		 * @return The pointer to the biome bin
		*/
		STPSingleHistogram::STPBin& operator[](Sample sample) {
			//check if the sample is in the dictionary
			const int diff = static_cast<int>(this->Dictionary.size() - sample);

			//get the biome using the index from dictionary
			//if not we need to insert that many extra entries so we can use sample to index the dictionary directly
			if (unsigned int& bin_index = diff <= 0
					? *this->Dictionary.insert_back_n(static_cast<size_t>((-diff) + 1u), NO_ENTRY)
					: this->Dictionary[sample];
				bin_index == NO_ENTRY) {
				//biome not exist, add and initialise
				STPSingleHistogram::STPBin& bin = this->Bin.emplace_back();
				bin.Item = sample;
				bin.Data.Quantity = 0u;
				//record the index in the bin and store to dictionary
				bin_index = static_cast<unsigned int>(this->Bin.size()) - 1;
				return bin;
			} else {
				return this->Bin[bin_index];
			}
		}

	public:

		STPAccumulator() = default;

		STPAccumulator(const STPAccumulator&) = delete;

		STPAccumulator(STPAccumulator&&) = delete;

		STPAccumulator& operator=(const STPAccumulator&) = delete;

		STPAccumulator& operator=(STPAccumulator&&) = delete;

		~STPAccumulator() = default;

		/**
		 * @brief Increment the sample bin by count
		 * @param sample The sample bin that will be operated on
		 * @param count The number to increment
		*/
		inline void inc(const Sample sample, const unsigned int count) {
			(*this)[sample].Data.Quantity += count;
		}

		/**
		 * @brief Decrement the sample bin by count
		 * If the bin will become empty after decrementing, bin will be erased from the accumulator and dictionary will
		 * be rehashed. Those operations are expensive, so don't call this function too often. In our implementation
		 * erasure rarely happens, benchmark shows this is still the best method.
		 * @param sample The sample bin that will be operated on
		 * @param count The number to decrement
		*/
		void dec(const Sample sample, const unsigned int count) {
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

	//After some experiment, we found out 4 parallel workers is the sweet spot.
	constexpr static unsigned char Parallelism = 4u;
	//A workplace is some available memory for a complete histogram generation
	typedef pair<STPDefaultHistogramBuffer, STPAccumulator> STPWorkplace;
	typedef unique_ptr<STPWorkplace[]> STPMemoryBlock;

	/**
	 * @brief STPMemoryBlockAllocator allocates a new memory block.
	*/
	struct STPMemoryBlockAllocator {
	public:

		inline STPMemoryBlock operator()() const {
			return make_unique<STPWorkplace[]>(STPSHFKernel::Parallelism);
		}

	};

	//All available workplaces are expressed as queue of pointers.
	STPObjectPool<STPMemoryBlock, STPMemoryBlockAllocator> MemoryBlockCache;
	//A multi-thread worker for concurrent per-pixel histogram generation
	STPThreadPool FilterWorker;

	/**
	 * @brief Copy the content in accumulator to the histogram buffer.
	 * Caller should make sure Output buffer has been preallocated, the size equals to the sum of all thread buffers.
	 * @tparam Normalise True to normalise the histogram in accumulator before copying.
	 * After normalisation, STPBin.Data should use Weight rather than Quantity,
	 * and the sum of weight of all bins in the accumulator is 1.0f.
	 * @param target The target histogram buffer.
	 * @param acc The accumulator to be copied.
	*/
	template<bool Normalise>
	static void copyToBuffer(STPDefaultHistogramBuffer& target, STPAccumulator& acc) {
		const auto [acc_beg, acc_end] = acc();
		target.HistogramStartOffset.emplace_back(static_cast<unsigned int>(target.Bin.size()));

		//instead of using the slow back_inserter, we can resize the array first
		const size_t bin_size = acc_end - acc_beg;
		target.Bin.resize(target.Bin.size() + bin_size);
		auto target_dest_begin = target.Bin.end() - bin_size;
		//copy bin
		if constexpr (Normalise) {
			//sum everything in the accumulator
			const float normFactor = 1.0f / static_cast<float>(std::accumulate(acc_beg, acc_end, 0u,
				[](const unsigned int init, const STPSingleHistogram::STPBin& bin) { return init + bin.Data.Quantity; }));
			std::transform(acc_beg, acc_end, target_dest_begin, [normFactor](auto bin) {
				//we need to make a copy
				bin.Data.Weight = bin.Data.Quantity * normFactor;
				return bin;
			});
		} else {
			//just copy the data
			std::copy(acc_beg, acc_end, target_dest_begin);
		}
	}

	/**
	 * @brief Perform vertical pass histogram filter
	 * @param sample_map The input sample map, usually it's biomemap.
	 * @param freeslip_rangeX The number of row pixel in the free-slip range.
	 * @param dimensionY The number of column pixel of the samplemap in one chunk.
	 * @param vertical_start_offset The vertical starting offset on the texture.
	 * The start offset should make the worker starts at the first y coordinate of the central texture.
	 * @param w_range Denotes the width start and end that will be computed by the current function call.
	 * The range should start from the halo (central image x index minus radius), and should use global index.
	 * The range end applies as well (central image x index plus dimension plus radius)
	 * @param workplace The pointer to the allocated working memory.
	 * @param radius The radius of the filter.
	*/
	static void filterVertical(const Sample* const sample_map, const unsigned int freeslip_rangeX, const unsigned int dimensionY,
		const unsigned int vertical_start_offset, const uvec2 w_range, STPWorkplace& workplace, const unsigned int radius) {
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
			STPSHFKernel::copyToBuffer<false>(target, acc);
			//generate histogram
			for (unsigned int j = 1u; j < dimensionY; j++) {
				//load one pixel to the bottom while unloading one pixel from the top
				acc.inc(sample_map[di], 1u);
				acc.dec(sample_map[ui], 1u);

				//copy the accumulator to buffer
				STPSHFKernel::copyToBuffer<false>(target, acc);

				//advance to the next central pixel
				di += freeslip_rangeX;
				ui += freeslip_rangeX;
			}

			//clear the accumulator
			acc.clear();
		}
	}

	/**
	 * @brief Merge buffers from each thread into a large chunk of output data.
	 * It will perform offset correction for HistogramStartOffset.
	 * @param buffer The histogram buffer that will be merged to.
	 * @param workplace_memory The pointer to the thread memory where the buffer will be copied from.
	 * @param workplaceID The ID of the workplace in the department. Note that threadID 0 doesn't require offset correction.
	 * @param output_base The base start index from the beginning of output container for each thread for bin and histogram offset.
	*/
	static void copyToOutput(STPPinnedHistogramBuffer* const buffer, const STPDefaultHistogramBuffer& workplace_memory,
		const unsigned char workplaceID, const uvec2 output_base) {
		auto offset_base_it = buffer->HistogramStartOffset.begin() + output_base.y;
		//caller should guarantee the output container has been allocated that many elements,
		//we don't need to allocate memory here

		//copy histogram offset
		if (workplaceID != 0u) {
			//do a offset correction first
			//no need to do that for thread 0 since offset starts at zero
			std::transform(workplace_memory.HistogramStartOffset.cbegin(), workplace_memory.HistogramStartOffset.cend(),
				offset_base_it,
				//get the starting index, so the current buffer connects to the previous buffer seamlessly
				[bin_base = output_base.x](const auto offset) { return bin_base + offset; });
		} else {
			//direct copy for threadID 0
			std::copy(workplace_memory.HistogramStartOffset.cbegin(),
				workplace_memory.HistogramStartOffset.cend(), offset_base_it);
		}

		//copy the bin
		std::copy(workplace_memory.Bin.cbegin(), workplace_memory.Bin.cend(), buffer->Bin.begin() + output_base.x);
	}

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
	static void filterHorizontal(STPPinnedHistogramBuffer* const histogram_input, const uvec2& dimension, const uvec2 h_range,
		STPWorkplace& workplace, const unsigned int radius) {
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
				//it will be a bit tricky for the last pixel in the histogram since that's the last iterator in start
				//offset array. we have emplaced one more offset at the end of HistogramStartOffset in Output to
				//indicate the size of the entire flatten Bin
				for (unsigned int bin_index = bin_start; bin_index < bin_end; bin_index++) {
					const STPSingleHistogram::STPBin& curr_bin = histogram_input->Bin[bin_index];
					acc.inc(curr_bin.Item, curr_bin.Data.Quantity);
				}
			}
			//copy the first pixel radius to buffer
			//we can start normalising data on the go, the accumulator is complete for this pixel
			STPSHFKernel::copyToBuffer<true>(target, acc);
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
				STPSHFKernel::copyToBuffer<true>(target, acc);

				//advance to the next central pixel
				ri += dimension.y;
				li += dimension.y;
			}

			//clear the accumulator before starting the next row
			acc.clear();
		}
	}

public:

	/**
	 * @brief Initialise a STPSHFKernel instance.
	*/
	STPSHFKernel() : FilterWorker(STPSHFKernel::Parallelism) {

	}

	STPSHFKernel(const STPSHFKernel&) = delete;

	STPSHFKernel(STPSHFKernel&&) = delete;

	STPSHFKernel& operator=(const STPSHFKernel&) = delete;

	STPSHFKernel& operator=(STPSHFKernel&&) = delete;

	~STPSHFKernel() = default;

	/**
	 * @brief Perform a complete histogram filter
	 * @param sample_map The input sample map for filter.
	 * @param nn_info The information about the nearest_neighbour logic applies to the samplemap.
	 * @param histogram_output The histogram buffer that will be used as buffer, and also output the final output
	 * @param central_chunk_index The local free-slip coordinate points to the central chunk.
	 * @param radius The radius of the filter
	*/
	void filter(const Sample* const sample_map, const STPNearestNeighbourInformation& nn_info,
		STPPinnedHistogramBuffer* const histogram_output, const uvec2 central_chunk_index, const unsigned int radius) {
		using std::future;
		using std::cref;
		using std::ref;

		future<void> workgroup[STPSHFKernel::Parallelism];
		//calculate central texture starting index
		const uvec2 &dimension = nn_info.MapSize,
			central_starting_coordinate = dimension * central_chunk_index;

		//request a working memory
		STPMemoryBlock memoryBlock = this->MemoryBlockCache.requestObject();

		auto sync_then_copy_to_output = [&filter_worker = this->FilterWorker,
			histogram_output, &workgroup, &memoryBlock]() -> void {
			size_t bin_total = 0u,
				offset_total = 0u;

			//sync working threads and get the total length of all buffers
			for (unsigned char w = 0u; w < STPSHFKernel::Parallelism; w++) {
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
			for (unsigned char w = 0u; w < STPSHFKernel::Parallelism; w++) {
				workgroup[w] = filter_worker.enqueue(STPSHFKernel::copyToOutput,
					histogram_output, cref(memoryBlock[w].first), w, base);

				//get the base index for the next worker, so each worker only copies buffer belongs to them to
				//independent location
				STPDefaultHistogramBuffer& curr_buffer = memoryBlock[w].first;
				//bin_base
				base.x += static_cast<unsigned int>(curr_buffer.Bin.size());
				//offset_base
				base.y += static_cast<unsigned int>(curr_buffer.HistogramStartOffset.size());
			}
			//sync
			for (unsigned char w = 0u; w < STPSHFKernel::Parallelism; w++) {
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
			//our loop in the filter ends at less than (no equal).
			//we had already make sure radius is an even number to ensure divisibility, also it's not too large to go out of memory bound.
			const unsigned int width_start = central_starting_coordinate.x - radius,
				total_width = dimension.x + 2u * radius,
				width_step = total_width / STPSHFKernel::Parallelism;
			uvec2 w_range(width_start, width_start + width_step);
			for (unsigned char w = 0u; w < STPSHFKernel::Parallelism; w++) {
				workgroup[w] = this->FilterWorker.enqueue(STPSHFKernel::filterVertical, cref(sample_map),
					nn_info.TotalMapSize.x, nn_info.MapSize.y, central_starting_coordinate.y, w_range,
					ref(memoryBlock[w]), radius);
				//increment
				w_range.x = w_range.y;
				if (w == STPSHFKernel::Parallelism - 2u) {
					//calculate the range for the last thread, to ensure all remaining columns are all done by the last thread
					w_range.y = width_start + total_width;
				} else {
					w_range.y += width_step;
				}
			}
			//sync get the total length of all buffers and copy buffer to output
			sync_then_copy_to_output();
		}
		//perform horizontal filter
		{
			//unlike vertical pass, we start from the first pixel of output from previous stage, and the output contains the halo histogram.
			//height start from 0, output buffer has the same height as each texture, and 2 * radius addition to the horizontal width as halos.
			const unsigned int height_step = dimension.y / STPSHFKernel::Parallelism;
			uvec2 h_range(0u, height_step);
			for (unsigned char w = 0u; w < STPSHFKernel::Parallelism; w++) {
				workgroup[w] = this->FilterWorker.enqueue(STPSHFKernel::filterHorizontal,
					histogram_output, cref(dimension), h_range, ref(memoryBlock[w]), radius);
				//increment range
				h_range.x = h_range.y;
				if (w == STPSHFKernel::Parallelism - 2u) {
					h_range.y = dimension.y;
				} else {
					h_range.y += height_step;
				}
			}
			//sync, do the same thing as what vertical did
			sync_then_copy_to_output();
		}

		//finished, return working memory
		this->MemoryBlockCache.returnObject(std::move(memoryBlock));
	}

};

/* Single Histogram Filter main class */

void STPSingleHistogramFilter::STPPinnedHistogramBufferDeleter::operator()(STPPinnedHistogramBuffer* const ptr) const {
	std::default_delete<STPPinnedHistogramBuffer>()(ptr);
}

STPSingleHistogramFilter::STPSingleHistogramFilter() : Kernel(make_unique<STPSHFKernel>()) {

}

STPSingleHistogramFilter::~STPSingleHistogramFilter() = default;

STPSingleHistogramFilter::STPHistogramBuffer_t STPSingleHistogramFilter::createHistogramBuffer() {
	//I hate to use `new` but unfortunately make_unique doesn't work with custom deleter...
	return STPHistogramBuffer_t(new STPPinnedHistogramBuffer());
}

STPSingleHistogram STPSingleHistogramFilter::operator()(const Sample*const  samplemap, const STPNearestNeighbourInformation& nn_info, 
	const STPHistogramBuffer_t& histogram_output, const unsigned int radius) {
	//do some simple runtime check
	//first make sure radius is an even number
	if (radius == 0u || (radius & 0x01u) != 0x00u) {
		throw STPException::STPBadNumericRange("radius should be an positive even number");
	}
	//second make sure radius is not larger than the free-slip range
	const uvec2 central_chunk_index = nn_info.ChunkNearestNeighbour / 2u;
	if (const uvec2 halo_size = central_chunk_index * nn_info.MapSize;
		halo_size.x < radius || halo_size.y < radius) {
		throw STPException::STPBadNumericRange("radius is too large and will overflow free-slip boundary");
	}

	//looks safe now, start the filter
	this->Kernel->filter(samplemap, nn_info, histogram_output.get(), central_chunk_index, radius);

	return STPSingleHistogramFilter::readHistogramBuffer(histogram_output);
}

STPSingleHistogram STPSingleHistogramFilter::readHistogramBuffer(const STPHistogramBuffer_t& buffer) noexcept {
	return STPSingleHistogram{ buffer->Bin.data(), buffer->HistogramStartOffset.data() };
}