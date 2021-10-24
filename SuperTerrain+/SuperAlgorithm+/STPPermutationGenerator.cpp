#include <SuperAlgorithm+/STPPermutationGenerator.h>
#include <memory>
#include <stdexcept>

#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Utility/Exception/STPInvalidEnvironment.h>

//Import definition
#include <SuperTerrain+/Utility/STPSmartDeviceMemory.tpp>

//CUDA Runtime
#include <cuda_runtime.h>

#include <glm/trigonometric.hpp>

//Initial table, will be shuffled later
static constexpr unsigned char INIT_TABLE[256] = {
	0,1,2,3,4,5,6,7,8,9,10,
	11,12,13,14,15,16,17,18,19,20,
	21,22,23,24,25,26,27,28,29,30,
	31,32,33,34,35,36,37,38,39,40,
	41,42,43,44,45,46,47,48,49,50,
	51,52,53,54,55,56,57,58,59,60,
	61,62,63,64,65,66,67,68,69,70,
	71,72,73,74,75,76,77,78,79,80,
	81,82,83,84,85,86,87,88,89,90,
	91,92,93,94,95,96,97,98,99,100,
	101,102,103,104,105,106,107,108,109,110,
	111,112,113,114,115,116,117,118,119,120,
	121,122,123,124,125,126,127,128,129,130,
	131,132,133,134,135,136,137,138,139,140,
	141,142,143,144,145,146,147,148,149,150,
	151,152,153,154,155,156,157,158,159,160,
	161,162,163,164,165,166,167,168,169,170,
	171,172,173,174,175,176,177,178,179,180,
	181,182,183,184,185,186,187,188,189,190,
	191,192,193,194,195,196,197,198,199,200,
	201,202,203,204,205,206,207,208,209,210,
	211,212,213,214,215,216,217,218,219,220,
	221,222,223,224,225,226,227,228,229,230,
	231,232,233,234,235,236,237,238,239,240,
	241,242,243,244,245,246,247,248,249,250,
	251,252,253,254,255
};

using std::copy;
using std::begin;
using std::end;
using std::shuffle;
using std::make_pair;

using namespace SuperTerrainPlus::STPCompute;

STPPermutationGenerator::STPPermutationGenerator(const STPEnvironment::STPSimplexNoiseSetting& simplex_setting) {
	if (!simplex_setting.validate()) {
		throw STPException::STPInvalidEnvironment("value range from simplex noise setting cannot be validated");
	}
	this->Gradient2DSize = simplex_setting.Distribution;

	//seed the engine
	STPPermutationRNG rng;
	rng.seed(simplex_setting.Seed);

	//we allocate memory in cpu and shuffle the table, then copy back to gpu
	//I was thinking about unified memory but we don't need the memory on host after the init process
	//so using pure device memory will be faster to access than unified one
	//allocation
	unsigned char PERMUTATION_HOST[512];
	//copy one... copy first
	copy(begin(INIT_TABLE), end(INIT_TABLE), PERMUTATION_HOST);
	//shuffle first, the two copy must be the same
	shuffle(PERMUTATION_HOST, PERMUTATION_HOST + 256, rng);
	//copy this the shuffled result
	copy(PERMUTATION_HOST, PERMUTATION_HOST + 256, PERMUTATION_HOST + 256);

	//now copy the host table to the device
	this->ManagedPermutation = STPSmartDeviceMemory::makeDevice<unsigned char[]>(512ull);
	this->Permutation = this->ManagedPermutation.get();
	STPcudaCheckErr(cudaMemcpy(this->Permutation, PERMUTATION_HOST, sizeof(unsigned char) * 512, cudaMemcpyHostToDevice));

	using glm::radians;
	//generate the gradient table
	//we are going to distribute the gradient evenly in a circle
	const double step = 360.0 / (this->Gradient2DSize * 1.0);//in degree
	std::unique_ptr<float[]> GRADIENT2D_HOST = std::make_unique<float[]>(this->Gradient2DSize * 2);//2D so we *2
	for (auto [angle, counter] = make_pair(0.0, 0u); angle < 360.0; angle += step, counter++) {//in degree
		const double offset_angle = angle + simplex_setting.Offset;
		GRADIENT2D_HOST[counter * 2] = static_cast<float>(glm::cos(radians(offset_angle)));
		GRADIENT2D_HOST[counter * 2 + 1] = static_cast<float>(glm::sin(radians(offset_angle)));
	}

	//copy the host gradient to device
	this->ManagedGradient2D = STPSmartDeviceMemory::makeDevice<float[]>(this->Gradient2DSize * 2);
	this->Gradient2D = this->ManagedGradient2D.get();
	STPcudaCheckErr(cudaMemcpy(this->Gradient2D, GRADIENT2D_HOST.get(), sizeof(float) * this->Gradient2DSize * 2, cudaMemcpyHostToDevice));
	//finishing up
}

const STPPermutation& STPPermutationGenerator::operator()() const {
	return dynamic_cast<const STPPermutation&>(*this);
}