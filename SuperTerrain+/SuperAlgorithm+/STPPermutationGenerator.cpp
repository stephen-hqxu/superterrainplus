#include <SuperAlgorithm+/STPPermutationGenerator.h>

//CUDA Runtime
#include <cuda_runtime.h>
//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

#include <glm/ext/scalar_constants.hpp>
#include <glm/trigonometric.hpp>

#include <array>
#include <limits>
#include <algorithm>
#include <memory>

using std::numeric_limits;
using std::array;

using std::copy;
using std::shuffle;
using std::make_pair;

using namespace SuperTerrainPlus::STPAlgorithm;

//The initial table contains numbers from 0 to max of unsigned char in order,
//this table will be shuffled into a random permutation in runtime.
constexpr static auto InitialTable = []() constexpr {
	array<unsigned char, numeric_limits<unsigned char>::max() + 1u> Table = { };

	for (unsigned short i = numeric_limits<unsigned char>::min(); i < Table.size(); i++) {
		Table[i] = static_cast<unsigned char>(i);
	}

	return Table;
}();

STPPermutationGenerator::STPPermutationGenerator(const STPEnvironment::STPSimplexNoiseSetting& simplex_setting) {
	simplex_setting.validate();
	this->Gradient2DSize = simplex_setting.Distribution;

	//seed the engine
	STPPermutationRNG rng;
	rng.seed(simplex_setting.Seed);

	//we allocate memory in cpu and shuffle the table, then copy back to gpu
	//I was thinking about unified memory but we don't need the memory on host after the init process
	//so using pure device memory will be faster to access than unified one
	//allocation
	array<unsigned char, InitialTable.size() * 2u> PermutationHost;
	//pointing to the beginning of the second half of the permutation
	auto PermutationHostHalfBegin = PermutationHost.begin() + InitialTable.size();

	copy(InitialTable.cbegin(), InitialTable.cend(), PermutationHost.begin());
	//shuffle first, the two copy must be the same
	shuffle(PermutationHost.begin(), PermutationHostHalfBegin, rng);
	//copy this the shuffled result
	copy(PermutationHost.begin(), PermutationHostHalfBegin, PermutationHostHalfBegin);

	//now copy the host table to the device
	this->ManagedPermutation = STPSmartDeviceMemory::makeDevice<unsigned char[]>(PermutationHost.size());
	this->Permutation = this->ManagedPermutation.get();
	STP_CHECK_CUDA(cudaMemcpy(this->Permutation, PermutationHost.data(),
		sizeof(unsigned char) * PermutationHost.size(), cudaMemcpyHostToDevice));

	constexpr static double TwoPi = 2.0 * glm::pi<double>();
	//generate the gradient table
	//we are going to distribute the gradient evenly in a circle
	const double step = TwoPi / (this->Gradient2DSize * 1.0);//in radians
	std::unique_ptr<float[]> Gradient2DHost = std::make_unique<float[]>(this->Gradient2DSize * 2);//2D so we *2
	for (auto [angle, counter] = make_pair(0.0, 0u); angle < TwoPi; angle += step, counter++) {//in degree
		const double offset_angle = angle + simplex_setting.Offset;
		Gradient2DHost[counter * 2] = static_cast<float>(glm::cos(offset_angle));
		Gradient2DHost[counter * 2 + 1] = static_cast<float>(glm::sin(offset_angle));
	}

	//copy the host gradient to device
	this->ManagedGradient2D = STPSmartDeviceMemory::makeDevice<float[]>(this->Gradient2DSize * 2);
	this->Gradient2D = this->ManagedGradient2D.get();
	STP_CHECK_CUDA(cudaMemcpy(this->Gradient2D, Gradient2DHost.get(), sizeof(float) * this->Gradient2DSize * 2, cudaMemcpyHostToDevice));
	//finishing up
}

const STPPermutation& STPPermutationGenerator::operator*() const {
	return dynamic_cast<const STPPermutation&>(*this);
}