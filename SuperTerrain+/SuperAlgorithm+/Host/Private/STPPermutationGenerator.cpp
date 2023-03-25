#include <SuperAlgorithm+Host/STPPermutationGenerator.h>
//Allocator
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceMemory.h>

//CUDA Runtime
#include <cuda_runtime.h>
//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

#include <glm/ext/scalar_constants.hpp>
#include <glm/trigonometric.hpp>

#include <array>
#include <limits>
#include <algorithm>
#include <utility>
#include <random>

using std::numeric_limits;
using std::array;
using std::integer_sequence;
using std::make_integer_sequence;

using std::copy;
using std::shuffle;
using std::make_pair;

using namespace SuperTerrainPlus::STPAlgorithm;

template<unsigned short... Seq>
constexpr static auto generateInitialTable(integer_sequence<unsigned short, Seq...>) {
	return array { static_cast<unsigned char>(Seq)... };
}

//The initial table contains numbers from 0 to max of unsigned char in order,
//this table will be shuffled into a random permutation in runtime.
//Use unsigned short to avoid overflow.
constexpr static auto InitialTable = generateInitialTable(make_integer_sequence<unsigned short, numeric_limits<unsigned char>::max() + 1u> {});

STPPermutationGenerator::STPPermutationTable STPPermutationGenerator::generate(const STPEnvironment::STPSimplexNoiseSetting& simplex_setting) {
	simplex_setting.validate();
	//distribution equals to size of gradient table
	const auto [seed, gradientSize, offset] = simplex_setting;

	//initialise a RNG
	std::mt19937_64 rng(seed);

	/* ----------------------------- generate permutation table ---------------------------- */
	//we allocate memory in host and shuffle the table, then copy back to device
	array<unsigned char, InitialTable.size() * 2u> permutationHost;
	//pointing to the beginning of the second half of the permutation
	const auto permutationHost_halfBeg = permutationHost.begin() + InitialTable.size();

	//copy the initial table to the memory
	copy(InitialTable.cbegin(), InitialTable.cend(), permutationHost.begin());
	//shuffle the initial table, and store the result to the memory
	shuffle(permutationHost.begin(), permutationHost_halfBeg, rng);
	//repeat this result to the second half
	copy(permutationHost.begin(), permutationHost_halfBeg, permutationHost_halfBeg);

	/* ------------------------------ generate gradient table ------------------------------ */
	constexpr static double TwoPi = 2.0 * glm::pi<double>();
	//we are going to distribute the gradient evenly in a circle
	const double step = TwoPi / (gradientSize * 1.0);//in radians
	//it's a 2D gradient table to we times 2
	const size_t gradientTableSize = gradientSize * 2u;

	const STPSmartDeviceMemory::STPHost<float[]> gradientHost = STPSmartDeviceMemory::makeHost<float[]>(gradientTableSize);
	for (auto [angle, counter] = make_pair(0.0, 0u); angle < TwoPi; angle += step, counter++) {
		const double offset_angle = angle + offset;
		gradientHost[counter * 2u] = static_cast<float>(glm::cos(offset_angle));
		gradientHost[counter * 2u + 1u] = static_cast<float>(glm::sin(offset_angle));
	}

	/* ---------------------------------- prepare output ---------------------------------- */
	STPPermutationTable result;
	auto& [result_d, result_h] = result;
	auto& [perm_device, grad_device] = result_d;

	//allocation
	perm_device = STPSmartDeviceMemory::makeDevice<unsigned char[]>(permutationHost.size());
	grad_device = STPSmartDeviceMemory::makeDevice<float[]>(gradientTableSize);
	//copy
	STP_CHECK_CUDA(cudaMemcpy(perm_device.get(), permutationHost.data(), sizeof(unsigned char) * permutationHost.size(), cudaMemcpyHostToDevice));
	STP_CHECK_CUDA(cudaMemcpy(grad_device.get(), gradientHost.get(), sizeof(float) * gradientTableSize, cudaMemcpyHostToDevice));
	
	//fill in permutation header data
	result_h = STPPermutation {
		perm_device.get(),
		grad_device.get(),
		gradientSize
	};
	return result;
}