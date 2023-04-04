#include <SuperTerrain+/Utility/STPDeviceLaunchSetup.cuh>

#include <glm/gtc/type_ptr.hpp>

//Standard Library
#include <cassert>
#include <cstdint>

#include <array>
#include <algorithm>
#include <iterator>
#include <limits>

using glm::uvec2, glm::uvec3;
using glm::value_ptr;

using std::array;
using std::pair, std::make_pair;
using std::sort, std::min_element, std::distance;

using namespace SuperTerrainPlus;
using STPDeviceLaunchSetup::STPDimensionSize;

namespace {

	/**
	 * @brief The rounding mode.
	*/
	enum class STPRoundMode : unsigned char {
		//Round to the previous power-of-2.
		//If the number itself is a power-of-2, do nothing.
		Previous = 0xAAu,
		//Default is round to nearest power-of-2.
		Default = 0xFFu
	};

}

//TODO: replace round to power-of-2 and nearest power-of-2 divisor (count trailing zeros) with the bit manipulation library in C++ 20

/**
 * @brief Bit Twiddling Hacks By Sean Eron Anderson to round the input decimal to an integer to a power-of-2.
 * http://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightMultLookup
 * @param Mode Specifies the rounding mode.
 * @param number The integer number input.
 * @return The rounded power-of-2 integer.
*/
template<STPRoundMode Mode>
inline static unsigned int roundToPowerOf2(const unsigned int number) noexcept {
	constexpr static auto shiftCount = array { 1u, 2u, 4u, 8u, 16u };

	unsigned int next = number;//next power of 2
	next--;
	std::for_each(shiftCount.cbegin(), shiftCount.cend(), [&next](const auto bit) { next |= next >> bit; });
	next++;
	const unsigned int prev = next >> 1u;//previous power of 2

	if constexpr (Mode == STPRoundMode::Previous) {
		return next == number ? next : prev;
	} else {
		return next - number > number - prev ? prev : next;
	}
}

/**
 * @brief Calculate a divisor closest to `b` that is divisible by `a` and it is power of 2.
 * The algorithm is provided by, also, Bit Twiddling Hacks.
 * @param a The number.
 * @param b The divisor. It is required that `b` is a power-of-2 integer.
 * @return The divisor as required, which is a power-of-2.
 * If `a` is less than `b`, return one. This is to avoid returning zero as divisor and cause undefined behaviour during integer division.
*/
inline static unsigned int calcNearestPowerOf2Divisor(const unsigned int a, const unsigned int b) noexcept {
	constexpr static unsigned int MultiplyDeBruijnBitPosition[] = { 0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17,
		4, 8, 31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9 };
	//Finding such divisor can be done by checking for divisibility.
	//A fast way of doing this if we know the divisor is a power-of-2 number is by keep right-shifting the divisor by 1 until we reach 0
	//	and then the equality `(a & (divisor - 1u)) == 0` if divisible.
	//This method essentially checks if all trailing bits of `a` is zero, but requires a loop.
	//Then we can derive a new formula based on this idea by finding the number of trailing zero of `a`.
	constexpr static auto countTrailingZero = [](const unsigned int n) constexpr noexcept -> unsigned int {
#pragma warning(push)
#pragma warning(disable: 4146)//use minus sign on unsigned number
		return MultiplyDeBruijnBitPosition[(static_cast<std::uint32_t>(n & -n) * 0x077CB531u) >> 27u];
#pragma warning(pop)
	};

	if (a <= b) {
		return 1u;
	}
	//since we know `b` is power-of-2, the number of trailing zero is simply the exponent, we can just clamp that
	const unsigned int exponent = std::min(countTrailingZero(a), countTrailingZero(b));
	return 1u << exponent;
}

/**
 * @brief Calculate the N root of an integer.
 * @param power The power of the root. Support range between 1 and 3 inclusive.
 * @param n The integer.
 * @return The N root of the integer.
*/
inline static double nrti(const STPDimensionSize power, const unsigned int n) noexcept {
	assert(power >= 2u && power <= 3u);

	if (power == 2u) {
		return std::sqrt(n);
	}
	//there is no need to support other powers because we don't need them
	return std::cbrt(n);
}

/**
 * @brief Determine the block size that minimises the number of idling thread.
 * The main idea is rectangle packing, given a large fixed size rectangle and a number of fixed area rectangles with unknown size.
 * @param blockDim The block dimension.
 * @param blockSize The original linear block size.
 * @param threadSize The thread size.
 * @return The optimal block size using different algorithms.
*/
static auto determineOptimalBlockSize(const STPDimensionSize blockDim, const unsigned int blockSize, uvec3 threadSize) {
	{
		unsigned int* const thread_begin = value_ptr(threadSize);
		unsigned int* const thread_end = thread_begin + 3u;
		//sort the thread size because we want to start checking from the smallest component
		sort(thread_begin, thread_end);
	}

	//the result from different block size calculation algorithm
	struct STPAlgorithmResult {
	public:

		uvec3 BlockDimension;

		//We will skip the last axis, and need to make sure the product of all axes of block dimension is the area.
		inline STPAlgorithmResult(const unsigned int blockSize) noexcept : BlockDimension(uvec3(uvec2(1u), blockSize)) {

		}

		~STPAlgorithmResult() = default;

		//The last dimension will be used to track the remaining area of the block.
		inline unsigned int& remainArea() noexcept {
			return this->BlockDimension[2];
		}

		//Update remaining block area and record block size for a particular axis.
		//Please do not record the last axis because this is used for tracking the total area.
		inline void recordAxis(const STPDimensionSize axis, const unsigned int size) noexcept {
			assert(axis >= 0u && axis <= 1u);
			this->remainArea() /= size;
			this->BlockDimension[axis] = size;
		}

		inline operator uvec3() const noexcept {
			return this->BlockDimension;
		}

	};
	STPAlgorithmResult bestFit(blockSize),
		minPerimeter = bestFit;

	//Start from the thread size axis with the smallest value.
	//Skip the first few starting axes for block dimensions that are unused, so the block size for these axes remain as one.
	for (STPDimensionSize dim = 3u - blockDim; dim < 2u; dim++) {
		//Round the thread extent to the previous even number; does not matter whether it is previous/nearest/next, it's just cheaper.
		//There is no way we can ensure this is divisible by power-of-2 if the extent is odd.
		const unsigned int thread_extent = threadSize[dim] & ~0x01u;

		/* -------------------------------------------- Best Fit Extent -------------------------------------------- */
		{
			//This algorithm tries to find a thread size that is small enough that the extent of the block may potentially fit perfectly.
			//Now find the block extent that fits the thread extent with minimal loss.
			//We want to ensure this block extent is an integer multiple of the remaining block area, and the thread extent.
			//While we cannot guarantee it will always be a multiple of thread extent,
			//	we want to maintain a multiple of block area;
			const unsigned int thread_multiple = calcNearestPowerOf2Divisor(thread_extent,
					roundToPowerOf2<STPRoundMode::Previous>(thread_extent)),
				block_multiple = calcNearestPowerOf2Divisor(bestFit.remainArea(), thread_multiple);
			bestFit.recordAxis(dim, block_multiple);
		}
		/* ------------------------------------------- Minimum Perimeter ------------------------------------------ */
		{
			//This algorithm attempts to determine the block size with minimum perimeter.
			//I don't know if it is a lemma or not, but seems like if a best fit extent is not found,
			//	a minimum perimeter block size actually fits very well.
			const unsigned int remain_area = minPerimeter.remainArea(),
				block_extent = calcNearestPowerOf2Divisor(remain_area,
					roundToPowerOf2<STPRoundMode::Default>(static_cast<unsigned int>(nrti(3u - dim, remain_area))));
			minPerimeter.recordAxis(dim, block_extent);
		}
	}

	//reorder the result to give a preference, in case when multiple algorithms give an identical and optimal result.
	return array<uvec3, 2u> { minPerimeter, bestFit };
}

/**
 * @brief Determine the launch configuration that gives the minimum number of idling thread.
 * This is done by rotating the block dimension and searching for an orientation that gives
 * the minimum number of total thread.
 * @param dimThread The thread dimension.
 * Unused dimension should be filled with 1.
 * @param dimBlock The block dimension.
 * @return The launch configuration with the minimum redundancy, and the actual number of thread used.
*/
static pair<STPDeviceLaunchSetup::STPLaunchConfiguration, std::uint64_t> determineMinimumRedundancyConfiguration(
	const uvec3& dimThread, uvec3 dimBlock) {
	using std::fill, std::next_permutation;
	using std::numeric_limits;
	constexpr static auto calcGridSize = [](const uvec3& blockSize, const uvec3& threadSize) constexpr noexcept -> uvec3 {
		return (threadSize + blockSize - 1u) / blockSize;
	};

	constexpr static size_t BlockAxisPermutation = 6u;//equals 3!
	//we shall rotate the block to the point when the number of idling thread is minimum
	//first by generating all possible permutations of the block size
	array<uvec3, BlockAxisPermutation> blockSizePermutation;
	array<std::uint64_t, BlockAxisPermutation> threadCountPermutation;
	//fill them with max value, so any unused slot will not be accidentally regarded as minimum
	fill(blockSizePermutation.begin(), blockSizePermutation.end(), uvec3(numeric_limits<unsigned int>::max()));
	fill(threadCountPermutation.begin(), threadCountPermutation.end(), numeric_limits<std::uint64_t>::max());

	unsigned int* const block_begin = value_ptr(dimBlock);
	unsigned int* const block_end = block_begin + 3u;
	//sort the block dimension
	sort(block_begin, block_end);
	{
		unsigned int i = 0u;
		do {
			//calculate the total number of thread in the current setup
			blockSizePermutation[i] = dimBlock;

			//use 64-bit integer because there is a danger of overflow with 32-bit
			const glm::u64vec3 actual_thread = dimBlock * calcGridSize(dimBlock, dimThread);
			threadCountPermutation[i] = actual_thread.x * actual_thread.y * actual_thread.z;
			i++;
		} while (next_permutation(block_begin, block_end));
	}
	//find the minimum thread count
	const auto min_thread_cout_it = min_element(threadCountPermutation.cbegin(), threadCountPermutation.cend());
	const ptrdiff_t min_idx = distance(threadCountPermutation.cbegin(), min_thread_cout_it);
	dimBlock = blockSizePermutation[min_idx];
	//round up the grid size
	const uvec3 dimGrid = calcGridSize(dimBlock, dimThread);

	return make_pair(make_pair(
		dim3(dimGrid.x, dimGrid.y, dimGrid.z),
		dim3(dimBlock.x, dimBlock.y, dimBlock.z)
	), *min_thread_cout_it);
}

__host__ STPDeviceLaunchSetup::STPLaunchConfiguration STPDeviceLaunchSetup::STPInternal::determineLaunchConfiguration(
	const STPDimensionSize blockDim, const unsigned int blockSize, const uvec3 threadSize) {
	const auto dimBlock = determineOptimalBlockSize(blockDim, blockSize, threadSize);
	constexpr static size_t BlockResultCount = dimBlock.size();
	
	//we want to calculate the configuration for each block size
	array<STPDeviceLaunchSetup::STPLaunchConfiguration, BlockResultCount> config;
	array<std::uint64_t, BlockResultCount> threadCount;
	for (size_t i = 0u; i < BlockResultCount; i++) {
		const auto [curr_config, curr_count] = determineMinimumRedundancyConfiguration(threadSize, dimBlock[i]);
		config[i] = curr_config;
		threadCount[i] = curr_count;
	}

	//now we want to find the configuration with the least thread count
	//if multiple of them are the same, pick the first minimum
	const ptrdiff_t min_idx = distance(threadCount.cbegin(),
		min_element(threadCount.cbegin(), threadCount.cend()));
	return config[min_idx];
}