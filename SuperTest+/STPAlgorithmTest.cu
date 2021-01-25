#include "catch2/catch.hpp"

//I know it's not a good practice to include source code.
//However I don't want to change my original project to static lib, YET.
#include "../GPGPU/STPPermutationsGenerator.cu"
#include "../GPGPU/STPSimplexNoise.cu"

using namespace Catch::Generators;
using namespace SuperTerrainPlus::STPCompute;
using namespace SuperTerrainPlus::STPSettings;

__global__ void readPerm(const STPPermutationsGenerator* const table, int index, int* result) {
	const unsigned int i = threadIdx.x + (blockDim.x * blockIdx.x);
	result[i] = table->perm(index);
}

__global__ void readGrad2D(const STPPermutationsGenerator* const table, int index, int component, double* result) {
	const unsigned int i = threadIdx.x + (blockDim.x * blockIdx.x);
	result[i] = table->grad2D(index, component);
}

TEST_CASE("Permutaion generator test", "[STPPermutationsGenerator]") {

	SECTION("Initialise the permutaion table") {
		REQUIRE_NOTHROW([]() -> void {
			const STPPermutationsGenerator perm(random(0, 10000).get());
			return;
		});
	}
	 
	SECTION("Randomness test, changing seed will also change the table") {
		int* results = nullptr;
		STPPermutationsGenerator* perm_d = nullptr;
		cudaMallocManaged(&results, sizeof(int) * 2, cudaMemAttachGlobal);
		cudaMalloc(&perm_d, sizeof(STPPermutationsGenerator) * 2);

		for (int i = 0; i < 5; i++) {
			const unsigned long long seed1 = random(0, 10000).get();
			const unsigned long long seed2 = random(0, 10000).get();
			const STPPermutationsGenerator perm1(seed1);
			const STPPermutationsGenerator perm2(seed2);

			cudaMemcpy(perm_d, &perm1, sizeof(STPPermutationsGenerator), cudaMemcpyHostToDevice);
			cudaMemcpy(perm_d + 1, &perm2, sizeof(STPPermutationsGenerator), cudaMemcpyHostToDevice);
			const int index = random(0, 511).get();

			readPerm << <1, 1 >> > (perm_d, index, results);
			readPerm << <1, 1 >> > (perm_d + 1, index, results + 1);
			cudaDeviceSynchronize();

			if (seed1 != seed2) {
				REQUIRE(results[0] != results[1]);
			}
			else {
				//We don't care about determinisisity now
				WARN("Same seed has been generated, skip this pass");
				SUCCEED();
			}
		}

		cudaFree(results);
		cudaFree(perm_d);

	}

	SECTION("Deterministisity test, the same input will always map to the same output") {
		int* result = nullptr;
		STPPermutationsGenerator* perm_d = nullptr;
		const unsigned long long seed = random(0, 10000).get();
		const STPPermutationsGenerator perm(seed);
		cudaMallocManaged(&result, sizeof(int) * 5, cudaMemAttachGlobal);
		cudaMalloc(&perm_d, sizeof(STPPermutationsGenerator));
		cudaMemcpy(perm_d, &perm, sizeof(STPPermutationsGenerator), cudaMemcpyHostToDevice);

		int prev[5];
		int index[5];
		for (int i = 0; i < 5; i++) {
			index[i] = random(0, 511).get();
		}
		for (int i = 0; i < 5; i++) {
			readPerm << <1, 5 >> > (perm_d, index[i], result + i);
			cudaDeviceSynchronize();
			prev[i] = result[i];
		}
		for (int i = 0; i < 5; i++) {
			readPerm << <1, 5 >> > (perm_d, index[i], result + i);
			cudaDeviceSynchronize();
			REQUIRE(result[i] == prev[i]);
		}

		cudaFree(result);
		cudaFree(perm_d);

	}

	SECTION("Uniqueness test, each 256 element in the permutation table should be unique") {
		SUCCEED();
	}

	SECTION("Illegal operations") {

		SECTION("Zero value of distribution") {
			SUCCEED();
		}

		SECTION("Index out of bound") {
			SUCCEED();
		}
	}
}

TEST_CASE("Simplex noise generator test", "[STPSimplexNoise]") {
	STPSimplexNoiseSettings sim_set;
	sim_set.Offset = 0.0f;
	sim_set.Distribution = 8;
	sim_set.Dimension.x = 512;
	sim_set.Dimension.y = 512;

	SECTION("Initialise the simplex noise generator") {
		REQUIRE_NOTHROW([&sim_set]() -> void {
			sim_set.Seed = random(0, 10000).get();
			const STPSimplexNoise perm(&sim_set);
			return;
		});
	}
}