#include <SuperAlgorithm+Host/STPSingleHistogramFilter.h>

//Memory
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceMemory.h>
//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>
#include <SuperTerrain+/Exception/STPNumericDomainError.h>
#include <SuperTerrain+/Exception/STPInvalidEnum.h>

//Utility
#include <limits>
#include <numeric>
#include <algorithm>
#include <type_traits>

#include <array>
#include <cstring>

using namespace SuperTerrainPlus::STPAlgorithm;
using SuperTerrainPlus::STPSample_t;

using glm::vec2;
using glm::uvec2;

using std::array;
using std::pair;
using std::unique_ptr;

using std::decay_t;

using std::make_pair;
using std::make_unique;
using std::visit;

//After some experiment, using 4 parallel workers is the sweet spot.
constexpr static size_t shfParallelism = 4u;

//TODO: use bit_cast instead in C++ 20
/**
 * @brief Perform a strict-aliasing conforming bitwise casting.
 * @tparam To The destination type.
 * @tparam From The source type.
 * @param source The value to be cast.
 * @return The destination value.
*/
template<typename To, typename From>
inline static To castBit(const From source) noexcept {
	using std::is_arithmetic;
	static_assert(std::conjunction_v<is_arithmetic<To>, is_arithmetic<From>> && sizeof(To) == sizeof(From),
		"The source and destination type must both be primitive and have the same size");
	//I really want to use `bit_cast` because `memcpy` is too ugly.
	//I know it is not portable, but most compilers use this function signature for C++20 `bit_cast`.
	//We will replace it with the proper standard library bit_cast soon anyway, this is for transition.
	return __builtin_bit_cast(To, source);
}

/* Implementation of custom data structure */
namespace {
	/**
	 * @brief A simple implementation of std::vector.
	 * After some profiling a custom data structure works better than the stl one due to its simplicity.
	 * @tparam T The data type of the array list.
	 * @param Pinned Specifies if the memory allocation should use a pinned memory allocator.
	*/
	template<class T, bool Pinned>
	class STPArrayList {
	public:

		//The iterator for the array list
		using iterator = T*;
		//The const iterator for the array list
		using const_iterator = const T*;

	private:

		static_assert(std::is_trivial_v<T>, "type must be a trivial type");
		static_assert(std::is_trivially_destructible_v<T>, "type must be trivially destructible");

		//The type of memory for the array list.
		typedef std::conditional_t<Pinned, SuperTerrainPlus::STPSmartDeviceMemory::STPHost<T[]>, unique_ptr<T[]>> STPMemory;
		
		//The data held by the array list.
		STPMemory Memory;

		//The beginning of the array.
		iterator Begin;
		//The pointer to the end of the array.
		iterator End;
		//The pointer to the end of the internal storage array
		iterator Last;

		/**
		 * @brief Expand the internal storage capacity
		 * @param new_capacity The new capacity
		*/
		void expand(size_t new_capacity) {
			//clamp the capacity, because a newly created array list might have zero capacity.
			new_capacity = std::max(static_cast<size_t>(1u), new_capacity);
			//the current size of the user array
			const size_t current_size = this->size();

			//allocate a cache
			STPMemory cache;
			if constexpr (Pinned) {
				cache = SuperTerrainPlus::STPSmartDeviceMemory::makeHost<T[]>(new_capacity);
			} else {
				cache = make_unique<T[]>(new_capacity);
			}

			//move data to the new memory
			if (current_size > 0u) {
				//there is a potential the pointer is null initially when there was no storage allocated
				//need to check to make sure, even if size is 0
				std::memcpy(cache.get(), this->Begin, current_size * sizeof(T));
			}
			//update our variables
			this->Memory = std::move(cache);
			this->Begin = this->Memory.get();
			this->End = this->Begin + current_size;
			this->Last = this->Begin + new_capacity;
		}

		/**
		 * @brief Insert count many elements at the end without initialisation
		 * @param count The number of element to be inserted
		*/
		inline void insert_back_n(const size_t count) {
			if (this->End + count > this->Last) {
				//not enough room, reallocation
				this->expand(std::max(this->capacity() * 2u, this->size() + count));
			}
			//simply move the end iterator ahead
			this->End += count;
		}

	public:

		inline STPArrayList() noexcept : Begin(nullptr), End(nullptr), Last(nullptr) {

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
		*/
		template<class... Arg>
		inline void emplace_back(Arg&&... arg) {
			this->insert_back_n(1u);

			//this is actually pretty dangerous.
			//if the type has non-trivial destructor and it will be called on the garbage data at the end iterator, which results in undefined behaviour.
			//but it's fine since we are dealing with trivial type only
			*(this->End - 1u) = T { std::forward<Arg>(arg)... };
		}

		/**
		 * @brief Insert count many elements at the end with initialisation
		 * @param count The number of element to be inserted
		 * @param value The value to be filled for every new element
		 * @return The iterator to the last inserted element
		*/
		iterator insert_back_n(const size_t count, const T& value) {
			this->insert_back_n(count);

			//initialise
			std::fill(this->End - count, this->End, value);
			return this->End - 1;
		}

		/**
		 * @brief Erase the item pointed by the iterator
		 * Erasure of an empty container is undefined behaviour
		 * @param it The item to be erased
		 * @return The iterator to the item following the iterator provided
		*/
		iterator erase(const iterator it) {
			//we don't need to call the destructor since T is always trivially destructible
			if (it < this->End - 1) {
				//it's not the last element, we need to move the memory forward
				//there's nothing to move if it's the last element, plus it may trigger undefined behaviour because end pointer should not be dereferenced
				//move_start is out of the range of [first, last) so we are good to use
				//no need to use std::move because T is known to be a PoD
				std::copy(it + 1, this->End, it);
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
				this->End = this->Begin + new_size;
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

		//Similarly, but a constant version.
		inline const T& operator[](const size_t index) const noexcept {
			return this->Begin[index];
		}

		/**
		 * @brief Clear the internal storage of the array list.
		 * Clear takes constant time as the array list only deals with trivial type
		*/
		inline void clear() noexcept {
			//since the array list only holds trivially destructible type, we can simply move the pointer
			this->End = this->Begin;
		}

		/**
		 * @brief Retrieve the number of element stored in the array list
		 * @return The number of element
		*/
		inline size_t size() const noexcept {
			return this->End - this->Begin;
		}

		/**
		 * @brief Get the number of element can be held by the internal storage
		 * @return The capacity
		*/
		inline size_t capacity() const noexcept {
			return this->Last - this->Begin;
		}

		/**
		 * @brief Get the constant iterator to the beginning of the array list
		 * @return The const iterator to the beginning of the array list
		*/
		inline const_iterator cbegin() const noexcept {
			return this->Begin;
		}

		/**
		 * @brief Get the constant iterator to the end of the array list
		 * @return The const iterator to the end of the array list
		*/
		inline const_iterator cend() const noexcept {
			return this->End;
		}

		/**
		 * @brief Get the iterator to the beginning of the array list
		 * @return The iterator to the beginning of the array list
		*/
		inline iterator begin() noexcept {
			return this->Begin;
		}

		/**
		 * @brief Get the iterator to the end of the array list
		 * @return The iterator to the end of the array list.
		 * Note that dereferencing this iterator will result in undefined behaviour.
		*/
		inline iterator end() noexcept {
			return this->End;
		}

		/**
		 * @brief Get the pointer to the first element in the array list.
		 * @return The pointer to the first element.
		 * If array list is empty, return nullptr
		*/
		inline const T* data() const noexcept {
			return this->Begin;
		}

		/**
		 * @brief Perform a raw copy from a source array list to the current instance.
		 * @tparam SrcT The element type of the source container.
		 * @param source The pointer to the source container. The entire range from the source will be copied.
		 * It is a undefined behaviour if the source is the same as the current instance.
		 * @param my_offset The offset into the current container where the copy should start.
		 * It is a undefined behaviour if size of the source container is greater than `end() - (begin() + offset)`.
		*/
		template<typename SrcT, bool SrcPin>
		inline void copyFrom(const STPArrayList<SrcT, SrcPin>& source, const size_t my_offset) noexcept {
			static_assert(sizeof(SrcT) == sizeof(T), "The element size of the source and the current array list are not the same");
			std::memcpy(this->Begin + my_offset, source.cbegin(), source.size() * sizeof(T));
		}

	};

	//The bin that uses un-normalised weight to store the quantity item in the bin.
	//Only used during filter execution, should normalise the quantity before passing the result to user.
	typedef STPSingleHistogram::STPGenericBin<unsigned int> STPUnnormalisedBin;
	//This allows us to copy between 2 bins using simple memory copy.
	static_assert(sizeof(STPUnnormalisedBin) == sizeof(STPSingleHistogram::STPBin),
		"The platform does not guarantee the normalised and un-normalised bin are in the same size");

	/**
	 * @brief STPHistogramBuffer resembles STPHistogram, unlike which, this is a compute buffer during generation instead of
	 * a data structure that can be easily used by external environment directly.
	 * @tparam Pinned True to use pinned memory allocator for the histogram buffer
	*/
	template<bool Pinned>
	struct STPHistogramBuffer {
	public:

		//All flatten bins in all histograms
		STPArrayList<STPSingleHistogram::STPBin, Pinned> Bin;
		//Get the bin starting index for a pixel in the flatten bin array
		STPArrayList<unsigned int, Pinned> HistogramStartOffset;

		STPHistogramBuffer() noexcept = default;

		~STPHistogramBuffer() = default;

		/**
		 * @brief Clear containers in histogram buffer.
		 * It doesn't guarantee to free up memory allocated inside, acting as memory pool which can be reused.
		*/
		inline void clear() noexcept {
			this->Bin.clear();
			this->HistogramStartOffset.clear();
		}

	};
	
	//Use during the filter operation for caching, therefore uses a cheaper page-free allocation.
	typedef STPHistogramBuffer<false> STPInternalHistogramBuffer;
	//Intended to be passed externally to the user, use a GPU-friendly page-locked allocation.
	typedef STPHistogramBuffer<true> STPExternalHistogramBuffer;

	/**
	 * @brief Accumulator acts as a cache for each row or column iteration.
	 * The accumulator for the next pixel equals to the accumulator from the previous pixel plus the left/down
	 * out-of-range radius pixel and minus the right/up in-range radius pixel.
	*/
	class STPAccumulator {
	private:

		typedef STPArrayList<STPUnnormalisedBin, false> STPBinArray;

	public:

		//Use Sample as index, find the index in Bin for this sample
		STPArrayList<unsigned int, false> Dictionary;
		//Store the number of element
		STPBinArray Bin;

		//Array of bins in accumulator
		typedef STPBinArray::const_iterator STPBinIterator;
		typedef pair<STPBinIterator, STPBinIterator> STPBinArrayRange;

	private:

		//Denotes there is no entry in the dictionary that corresponds to the index in bin
		constexpr static unsigned int NoEntry = std::numeric_limits<unsigned int>::max();

	public:

		STPAccumulator() = default;

		~STPAccumulator() = default;

		/**
		 * @brief Increment the sample bin by count
		 * @param sample The sample bin that will be operated on
		 * @param count The number to increment
		*/
		void increment(const STPSample_t sample, const unsigned int count) {
			//Get the bin for a biome sample.
			//If bin doesn't exist, a new bin is created.
			//check if the sample is in the dictionary
			const int diff = static_cast<int>(this->Dictionary.size()) - static_cast<int>(sample);

			//get the biome using the index from dictionary
			//if not we need to insert that many extra entries so we can use sample to index the dictionary directly
			if (unsigned int& bin_index = diff <= 0
					? *this->Dictionary.insert_back_n(static_cast<size_t>((-diff) + 1u), STPAccumulator::NoEntry)
					: this->Dictionary[sample];
				bin_index == STPAccumulator::NoEntry) {
				//biome not exist, add and initialise
				this->Bin.emplace_back(sample, count);
				//record the index in the bin and store to dictionary
				bin_index = static_cast<unsigned int>(this->Bin.size() - 1u);
			} else {
				//increment count
				this->Bin[bin_index].Weight += count;
			}
		}

		/**
		 * @brief Decrement the sample bin by count
		 * If the bin will become empty after decrementing, bin will be erased from the accumulator and dictionary will
		 * be rehashed. Those operations are expensive, so don't call this function too often. In our implementation
		 * erasure rarely happens, benchmark shows this is still the best method.
		 * @param sample The sample bin that will be operated on
		 * @param count The number to decrement
		*/
		void decrement(const STPSample_t sample, const unsigned int count) {
			//our algorithm guarantees the bin has been increment by this sample before, so no check is needed
			unsigned int& bin_index = this->Dictionary[sample];
			unsigned int& quant = this->Bin[bin_index].Weight;

			if (quant <= count) {
				//update the dictionary entries linearly, basically it's a rehash in hash table
				auto followed_index = this->Bin.erase(this->Bin.begin() + bin_index);
				std::for_each(followed_index, this->Bin.end(), [&dic = this->Dictionary](const auto& bin) {
					//since all subsequent indices followed by the erased bin has been advanced forward by one block
					//we need to subtract the indices recorded in dictionary for those entries by one.
					dic[bin.Item]--;
				});

				//bin will become empty, erase this bin and dictionary entry
				bin_index = STPAccumulator::NoEntry;
			} else {
				//simply decrement the quantity otherwise
				quant -= count;
			}
		}

		/**
		 * @brief Get the array of bins in the accumulator, this is useful for doing fast memory copy
		 * @return Number of element in the bin and the pointer to the start of the bin array
		*/
		inline STPBinArrayRange iterator() const noexcept {
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
}

/* Single Histogram Filter implementation */

/**
 * @brief Copy the content in accumulator to the histogram buffer.
 * Caller should make sure Output buffer has been preallocated, the size equals to the sum of all thread buffers.
 * @tparam Normalise True to normalise the histogram in accumulator before copying.
 * After normalisation, STPBin.Data should use Weight rather than Quantity,
 * and the sum of weight of all bins in the accumulator is 1.0f.
 * @param BufPin Specifies if the target buffer uses pinned memory.
 * @param buffer The target histogram buffer.
 * @param acc The accumulator to be copied.
*/
template<bool Normalise, bool BufPin>
static void copyToBuffer(STPHistogramBuffer<BufPin>& buffer, STPAccumulator& acc) {
	buffer.HistogramStartOffset.emplace_back(static_cast<unsigned int>(buffer.Bin.size()));

	const size_t bin_size = acc.Bin.size(),
		//the original starting offset of the destination bin
		buffer_dest_offset = buffer.Bin.size();
	buffer.Bin.resize(buffer_dest_offset + bin_size);
	//copy bin
	if constexpr (Normalise) {
		const auto [acc_beg, acc_end] = acc.iterator();
		//sum everything in the accumulator
		const float normFactor = 1.0f / static_cast<float>(std::accumulate(acc_beg, acc_end, 0u,
			[](const auto init, const auto& bin) { return init + bin.Weight; }));
		//normalisation
		std::transform(acc_beg, acc_end, buffer.Bin.begin() + buffer_dest_offset, [normFactor](const auto& bin) {
			return STPSingleHistogram::STPBin { bin.Item, bin.Weight * normFactor };
		});
	} else {
		//just copy the data, we have to use the low-level copy because they are not the same type
		//we will make sure their sizes are the same
		buffer.Bin.copyFrom(acc.Bin, buffer_dest_offset);
	}
}

/**
 * @brief Perform vertical pass histogram filter
 * @param sample_map The input sample map, usually it's biomemap.
 * @param histogram_output The location where the output histogram will be stored.
 * @param worker_cache The intermediate cache memory for the current worker.
 * @param nn_rangeX The number of total row pixel in the nearest neighbour range.
 * @param dimensionY The number of column pixel of the samplemap in one chunk.
 * @param vertical_start_offset The vertical starting offset on the texture.
 * The start offset should make the worker starts at the first y coordinate of the central texture.
 * @param w_range Denotes the width start and end that will be computed by the current function call.
 * The range should start from the halo (central image x index minus radius), and should use global index.
 * The range end applies as well (central image x index plus dimension plus radius)
 * @param radius The radius of the filter.
*/
static void filterVertical(const STPSample_t* const sample_map, STPInternalHistogramBuffer& histogram_output,
	STPAccumulator& worker_cache, const unsigned int nn_rangeX, const unsigned int dimensionY,
	const unsigned int vertical_start_offset, const uvec2 w_range, const unsigned int radius) {
	//clear both
	histogram_output.clear();
	worker_cache.clear();

	const int radius_signed = static_cast<int>(radius);
	//we assume the radius never goes out of the nearest neighbour boundary
	//we are traversing the a row-major sample_map column by column
	for (unsigned int i = w_range.x; i < w_range.y; i++) {
		//the target (current) pixel index
		const unsigned int ti = i + vertical_start_offset * nn_rangeX;
		//the pixel index of up-most radius (inclusive of the current radius)
		unsigned int ui = ti - radius * nn_rangeX,
			//the pixel index of down-most radius (exclusive of the current radius)
			di = ti + (radius + 1u) * nn_rangeX;

		//load the radius into accumulator
		for (int j = -radius_signed; j <= radius_signed; j++) {
			worker_cache.increment(sample_map[ti + j * nn_rangeX], 1u);
		}
		//copy the first pixel to buffer
		copyToBuffer<false>(histogram_output, worker_cache);
		//generate histogram
		for (unsigned int j = 1u; j < dimensionY; j++) {
			//load one pixel to the bottom while unloading one pixel from the top
			worker_cache.increment(sample_map[di], 1u);
			worker_cache.decrement(sample_map[ui], 1u);

			//copy the accumulator to buffer
			copyToBuffer<false>(histogram_output, worker_cache);

			//advance to the next central pixel
			di += nn_rangeX;
			ui += nn_rangeX;
		}

		//clear the accumulator
		worker_cache.clear();
	}
}

/**
 * @brief Merge buffers from each thread into a large chunk of output data.
 * It will perform offset correction for HistogramStartOffset.
 * @param buffer_input The pointer to the thread memory where the buffer will be copied from.
 * @param buffer_outout The histogram buffer that will be merged to.
 * @param workerID The ID of the parallel worker. Note that ID of 0 doesn't require offset correction.
 * @param output_base The base start index from the beginning of output container for each thread for bin and histogram offset.
*/
static void copyToOutput(const STPInternalHistogramBuffer& buffer_input, STPExternalHistogramBuffer& buffer_output,
	const size_t workerID, const uvec2 output_base) {
	//caller should guarantee the output container has been allocated that many elements,
	//we don't need to allocate memory here

	//copy histogram offset
	if (workerID != 0u) {
		const auto offset_base_it = buffer_output.HistogramStartOffset.begin() + output_base.y;
		//do a offset correction first
		//no need to do that for thread 0 since offset starts at zero
		std::transform(buffer_input.HistogramStartOffset.cbegin(), buffer_input.HistogramStartOffset.cend(),
			offset_base_it,
			//get the starting index, so the current buffer connects to the previous buffer seamlessly
			[bin_base = output_base.x](const auto offset) { return bin_base + offset; });
	} else {
		//direct copy for threadID 0
		buffer_output.HistogramStartOffset.copyFrom(buffer_input.HistogramStartOffset, output_base.y);
	}

	//copy the bin
	buffer_output.Bin.copyFrom(buffer_input.Bin, output_base.x);
}

/**
 * @brief Perform horizontal pass histogram filter.
 * The input is the ouput from horizontal pass.
 * @param histogram_input The input histogram buffer from the vertical pass.
 * @param histogram_output The memory where the histogram for the horizontal pass will be stored.
 * @param worker_cache The caching memory location for intermediate computation result for the current worker.
 * @param dimension The dimension of one texture.
 * @param h_range Denotes the height start and end that will be computed by the current function call.
 * The range should start from 0.
 * The range end at the height of the texture.
 * @param radius The radius of the filter.
*/
template<bool InPin, bool OutPin>
static void filterHorizontal(const STPHistogramBuffer<InPin>& histogram_input,
	STPHistogramBuffer<OutPin>& histogram_output, STPAccumulator& worker_cache, const uvec2& dimension,
	const uvec2 h_range, const unsigned int radius) {
	//make sure both of them are cleared (don't deallocate)
	histogram_output.clear();
	worker_cache.clear();

	const int radius_signed = static_cast<int>(radius);
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
		for (int j = -radius_signed; j <= radius_signed; j++) {
			auto bin_offset = histogram_input.HistogramStartOffset.cbegin() + (ti + j * dimension.y);
			const unsigned int bin_start = *bin_offset,
				bin_end = *(bin_offset + 1);
			//it will be a bit tricky for the last pixel in the histogram since that's the last iterator in start
			//offset array. we have emplaced one more offset at the end of HistogramStartOffset in Output to
			//indicate the size of the entire flatten Bin
			for (unsigned int bin_index = bin_start; bin_index < bin_end; bin_index++) {
				const STPSingleHistogram::STPBin& curr_bin = histogram_input.Bin[bin_index];
				worker_cache.increment(curr_bin.Item, castBit<unsigned int>(curr_bin.Weight));
			}
		}
		//copy the first pixel radius to buffer
		//we can start normalising data on the go, the accumulator is complete for this pixel
		copyToBuffer<true>(histogram_output, worker_cache);
		//generate histogram, starting from the second pixel, we only loop through the central texture
		for (unsigned int j = 1u; j < dimension.x; j++) {
			//load one pixel to the right while unloading one pixel from the left
			auto bin_beg = histogram_input.HistogramStartOffset.cbegin();
			auto bin_offset_r = bin_beg + ri,
				bin_offset_l = bin_beg + li;
			//collect histogram at the right pixel
			{
				const unsigned int bin_start = *bin_offset_r,
					bin_end = *(bin_offset_r + 1);
				for (unsigned int bin_index = bin_start; bin_index < bin_end; bin_index++) {
					const STPSingleHistogram::STPBin& curr_bin = histogram_input.Bin[bin_index];
					worker_cache.increment(curr_bin.Item, castBit<unsigned int>(curr_bin.Weight));
				}
			}
			//discard histogram at the left pixel
			{
				const unsigned int bin_start = *bin_offset_l,
					bin_end = *(bin_offset_l + 1);
				for (unsigned int bin_index = bin_start; bin_index < bin_end; bin_index++) {
					const STPSingleHistogram::STPBin& curr_bin = histogram_input.Bin[bin_index];
					worker_cache.decrement(curr_bin.Item, castBit<unsigned int>(curr_bin.Weight));
				}
			}

			//copy accumulator to buffer
			copyToBuffer<true>(histogram_output, worker_cache);

			//advance to the next central pixel
			ri += dimension.y;
			li += dimension.y;
		}

		//clear the accumulator before starting the next row
		worker_cache.clear();
	}
}

/**
 * @brief Correct the histogram offset array by inserting the total number of bin at the end.
 * this is because during calculation, each pixel needs to access the current offset, and the offset of the next pixel,
 * insert one at the back to make sure the last pixel can read some valid data.
 * @param histogram The histogram to be corrected.
*/
template<bool Pin>
inline static void correctHistogramOffset(STPHistogramBuffer<Pin>& histogram) {
	histogram.HistogramStartOffset.emplace_back(static_cast<unsigned int>(histogram.Bin.size()));
}

/* STPFilterBuffer */

template<STPSingleHistogramFilter::STPFilterBuffer::STPExecutionType Exec>
struct STPSingleHistogramFilter::STPFilterBuffer::STPBufferMemory {
public:

	typedef pair<STPInternalHistogramBuffer, STPAccumulator> STPWorkplace;

	//True if the buffer memory is multithread capable.
	constexpr static STPFilterBuffer::STPExecutionType ExecutionType = Exec;

	//This is where all intermediate work stays, each worker gets their own intermediate memory space.
	std::conditional_t<Exec == STPExecutionType::Parallel,
		array<STPBufferMemory::STPWorkplace, shfParallelism>,
		STPBufferMemory::STPWorkplace> WorkingMemory;
	//This is where the filter output resides at.
	STPExternalHistogramBuffer OutputHistogram;

};

STPSingleHistogramFilter::STPFilterBuffer::STPFilterBuffer(const STPFilterBuffer::STPExecutionType execution_type) {
	switch (execution_type) {
	case STPExecutionType::Serial:
		this->Memory = make_unique<STPSerialBufferMemory>();
		break;
	case STPExecutionType::Parallel:
		this->Memory = make_unique<STPParallelBufferMemory>();
		break;
	default:
		throw STP_INVALID_ENUM_CREATE(execution_type, STPFilterBuffer::STPExecutionType);
	}
}

STPSingleHistogramFilter::STPFilterBuffer::STPFilterBuffer(STPFilterBuffer&&) noexcept = default;

STPSingleHistogramFilter::STPFilterBuffer& STPSingleHistogramFilter::STPFilterBuffer::operator=(STPFilterBuffer&&) noexcept = default;

STPSingleHistogramFilter::STPFilterBuffer::~STPFilterBuffer() = default;

inline const auto& STPSingleHistogramFilter::STPFilterBuffer::getOutput() const {
	return visit([](const auto& memory) -> STPExternalHistogramBuffer& { return memory->OutputHistogram; }, this->Memory);
}

STPSingleHistogram STPSingleHistogramFilter::STPFilterBuffer::readHistogram() const {
	const STPExternalHistogramBuffer& histogram = this->getOutput();
	return STPSingleHistogram { histogram.Bin.data(), histogram.HistogramStartOffset.data() };
}

STPSingleHistogramFilter::STPFilterBuffer::STPHistogramSize STPSingleHistogramFilter::STPFilterBuffer::size() const {
	const STPExternalHistogramBuffer& histogram = this->getOutput();
	return make_pair(histogram.Bin.size(), histogram.HistogramStartOffset.size());
}

STPSingleHistogramFilter::STPFilterBuffer::STPExecutionType
	STPSingleHistogramFilter::STPFilterBuffer::type() const noexcept {
	//TODO: use template lambda in C++ 20 as well
	return visit(
		[](const auto& memory) { return decay_t<decltype(memory)>::element_type::ExecutionType; }, this->Memory);
}

/* Single Histogram Filter main class */

STPSingleHistogramFilter::STPSingleHistogramFilter() : FilterWorker(shfParallelism) {

}

void STPSingleHistogramFilter::filter(const STPFilterKernelData& data) {
	const auto& [sample_map, nn_info, filter_memory, radius, vertical_start_coord, width_range, height_range] = data;
	using std::ref;

	const uvec2& dimension = nn_info.MapSize;
	//the filter memory might be single or multi-thread capable memory
	//TODO: use template lambda in C++ 20
	auto [memoryBlock, histogramOutput] = visit([](const auto& memory) noexcept {
		typedef decay_t<decltype(memory)>::element_type BufferMemory;
		BufferMemory::STPWorkplace* workplace;

		if constexpr (BufferMemory::ExecutionType == STPFilterBuffer::STPExecutionType::Parallel) {
			//we use the first working memory
			workplace = &memory->WorkingMemory[0];
		} else {
			workplace = &memory->WorkingMemory;
		}
		return make_pair(ref(*workplace), ref(memory->OutputHistogram));
	}, filter_memory.Memory);
	auto& [intermediate, cache] = memoryBlock;
	
	//similar to the distributed filter, we run the vertical pass first
	filterVertical(sample_map, intermediate, cache, nn_info.TotalMapSize.x, dimension.y, vertical_start_coord, width_range, radius);
	correctHistogramOffset(intermediate);

	//unlike distributed filter, however, we don't need to merge buffer, because there is only one working thread
	//we can start horizontal pass straight away
	filterHorizontal(intermediate, histogramOutput, cache, dimension, height_range, radius);
	correctHistogramOffset(histogramOutput);
}

void STPSingleHistogramFilter::filterDistributed(const STPFilterKernelData& data) {
	const auto& [sample_map, nn_info, filter_memory, radius, vertical_start_coord, width_range, height_range] = data;
	using std::array;
	using std::future;
	using std::get;
	using std::ref;
	using std::cref;

	const uvec2& dimension = nn_info.MapSize;
	auto& [memoryBlock, histogramOutput] = *get<unique_ptr<STPFilterBuffer::STPParallelBufferMemory>>(filter_memory.Memory);

	//wait for all filter workers to finish, then merge their results to one output
	//the parallel version of the filter assume the problem size is big so the number iteration is greater than the thread count
	const auto waitForWorker = [&filter_worker = this->FilterWorker,
		&memoryBlock, &histogramOutput](array<future<void>, shfParallelism>& filterTask) -> void {
		const size_t taskSize = filterTask.size();
		size_t bin_total = 0u, offset_total = 0u;

		//sync working threads and get the total length of all buffers
		for (size_t w = 0u; w < taskSize; w++) {
			filterTask[w].get();

			//grab the buffer in the allocated working memory belongs to this thread
			STPInternalHistogramBuffer& curr_buffer = memoryBlock[w].first;
			bin_total += curr_buffer.Bin.size();
			offset_total += curr_buffer.HistogramStartOffset.size();
		}

		//copy thread buffer to output
		//we don't need to clear the output, but rather we can resize it (items will get overwritten anyway)
		histogramOutput.Bin.resize(bin_total);
		histogramOutput.HistogramStartOffset.resize(offset_total);
		uvec2 base(0u);
		for (size_t w = 0u; w < taskSize; w++) {
			//get the base index for the next worker, so each worker only copies buffer belongs to them to independent location
			STPInternalHistogramBuffer& curr_buffer = memoryBlock[w].first;

			filterTask[w] = filter_worker.enqueue(copyToOutput, cref(curr_buffer), ref(histogramOutput), w, base);

			//bin_base
			base.x += static_cast<unsigned int>(curr_buffer.Bin.size());
			//offset_base
			base.y += static_cast<unsigned int>(curr_buffer.HistogramStartOffset.size());
		}
		//sync
		for (auto& block : filterTask) {
			block.get();
		}
		correctHistogramOffset(histogramOutput);
	};

	//we perform vertical first as the input sample_map is a row-major matrix, after this stage the output buffer will be in column-major.
	//then we perform horizontal filter on the output, so the final histogram will still be in row-major.
	//It seems like breaking the cache locality on sample_map but benchmark shows there's no difference in performance.

	//perform vertical filter
	{
		auto verticalBlock = this->FilterWorker.enqueueLoop<shfParallelism>(
			[sample_map, total_sizeX = nn_info.TotalMapSize.x, map_sizeY = dimension.y, vertical_start_coord,
				&memoryBlock, radius](const auto block_idx, const auto begin, const auto end) {
				auto& [output, cache] = memoryBlock[block_idx];
				filterVertical(sample_map, output, cache, total_sizeX, map_sizeY, vertical_start_coord, uvec2(begin, end), radius);
			}, width_range.x, width_range.y);
		//sync get the total length of all buffers and copy buffer to output
		waitForWorker(get<1u>(verticalBlock));
	}
	//perform horizontal filter
	{
		//unlike vertical pass, we start from the first pixel of output from previous stage, and the output contains the halo histogram.
		//height start from 0, output buffer has the same height as each texture, and 2 * radius addition to the horizontal width as halos.
		auto horizontalBlock = this->FilterWorker.enqueueLoop<shfParallelism>(
			[&histogramOutput, &dimension, &memoryBlock, radius](
				const auto block_idx, const auto begin, const auto end) {
				auto& [output, cache] = memoryBlock[block_idx];
				filterHorizontal(histogramOutput, output, cache, dimension, uvec2(begin, end), radius);
			}, height_range.x, height_range.y);
		//sync, do the same thing as what vertical did
		waitForWorker(get<1u>(horizontalBlock));
	}
}

STPSingleHistogram STPSingleHistogramFilter::operator()(const STPSample_t* const samplemap,
	const STPNearestNeighbourInformation& nn_info, STPFilterBuffer& filter_buffer, const unsigned int radius) {
	//do some simple runtime check
	//first make sure radius is an even number
	STP_ASSERTION_NUMERIC_DOMAIN(radius > 0u && (radius & 0x01u) == 0x00u, "radius should be an positive even number");
	//second make sure radius is not larger than the nearest neighbour range
	const uvec2& dimension = nn_info.MapSize;
	const uvec2 central_chunk_index = nn_info.ChunkNearestNeighbour / 2u,
		start_coord = dimension * central_chunk_index;
	STP_ASSERTION_NUMERIC_DOMAIN(radius <= start_coord.x && radius <= start_coord.y,
		"radius is too large and will overflow nearest neighbour boundary");
	
	//we need to start from the left halo, and ends at the right halo, width of which is radius.
	//our loop in the filter ends at less than (no equal).
	//we had already make sure radius is an even number to ensure divisibility, also it's not too large to go out of memory bound.
	const unsigned int width_start = start_coord.x - radius,
		width_end = width_start + dimension.x + 2u * radius;
	//looks safe now, start the filter
	const STPSingleHistogramFilter::STPFilterKernelData filterData {
		samplemap, nn_info, filter_buffer, radius,
		start_coord.y, uvec2(width_start, width_end), uvec2(0u, dimension.y)
	};
	//choose filter procedure depends on the filter buffer type
	if (filter_buffer.type() == STPFilterBuffer::STPExecutionType::Parallel
		&& dimension.x >= shfParallelism && dimension.y >= shfParallelism) {
		//also need to make sure the map size is big enough so works can be distributed to that many thread
		this->filterDistributed(filterData);
	} else {
		this->filter(filterData);
	}

	return filter_buffer.readHistogram();
}