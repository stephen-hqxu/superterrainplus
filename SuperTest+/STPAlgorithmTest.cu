#include "catch2/catch.hpp"

//I know it's not a good practice to include source code.
//However I don't want to change my original project to static lib, YET.
#include "../GPGPU/STPPermutationsGenerator.cu"
#include "../GPGPU/STPSimplexNoise.cu"

using namespace Catch::Generators; 
using Catch::Matchers::EndsWith;
using Catch::Matchers::Contains;

using namespace SuperTerrainPlus::STPCompute;
using namespace SuperTerrainPlus::STPSettings;

__global__ void readPerm(const STPPermutationsGenerator* const table, int index, int* result) {
	const unsigned int i = threadIdx.x + (blockDim.x * blockIdx.x);
	result[i] = table->perm(index);
}

__global__ void readPerm(const STPPermutationsGenerator* const table, int* result) {
	const unsigned int i = threadIdx.x + (blockDim.x * blockIdx.x);
	result[i] = table->perm(i);
}

__global__ void sim2d(const STPSimplexNoise* const noise, float2 coord, float* result) {
	const unsigned int i = threadIdx.x + (blockDim.x * blockIdx.x);
	result[i] = noise->simplex2D(coord.x, coord.y);
}

TEST_CASE("Permutaion generator test", "[STPPermutationsGenerator]") {

	SECTION("Initialise the permutaion table") {
		REQUIRE_NOTHROW(STPPermutationsGenerator(random(0, 10000).get()));
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
			readPerm << <1, 1 >> > (perm_d, index[i], result + i);
			cudaDeviceSynchronize();
			prev[i] = result[i];
		}
		for (int i = 0; i < 5; i++) {
			readPerm << <1, 1 >> > (perm_d, index[i], result + i);
			cudaDeviceSynchronize();
			INFO("The value was " << result[i] << " at index " << i << ", but got " << prev[i]);
			REQUIRE(result[i] == prev[i]);
		}

		cudaFree(result);
		cudaFree(perm_d);

	}

	SECTION("Uniqueness test, each 256 element in the permutation table should be unique") {
		int* result = nullptr;
		STPPermutationsGenerator* perm_d = nullptr;
		const unsigned long long seed = random(0, 10000).get();
		const STPPermutationsGenerator perm(seed);
		cudaMallocManaged(&result, sizeof(int) * 256, cudaMemAttachGlobal);
		cudaMalloc(&perm_d, sizeof(STPPermutationsGenerator));
		cudaMemcpy(perm_d, &perm, sizeof(STPPermutationsGenerator), cudaMemcpyHostToDevice);

		for (int i = 0; i < 5; i++) {
			readPerm << <1, 256 >> > (perm_d, result);
			cudaDeviceSynchronize();
			auto it = std::unique(result, result + 256);
			INFO(*it << " appears the second time");
			REQUIRE(it == result + 256);
		}
		
		cudaFree(result);
		cudaFree(perm_d);
	}

	SECTION("Illegal operations") {

		SECTION("Zero value of distribution") {
			REQUIRE_THROWS_WITH(STPPermutationsGenerator(random(0, 10000).get(), 0u, 0.0), Contains("0"));
		}

		SECTION("Index out of bound") {
			SUCCEED("Index out of bound will cause undefined behavior thus untestable");
		}
	}
}

TEST_CASE("Simplex noise generator test", "[STPSimplexNoise][!mayfail]") {
	STPSimplexNoiseSettings sim_set;
	sim_set.Offset = 0.0f;
	sim_set.Distribution = 8;
	sim_set.Dimension.x = 512;
	sim_set.Dimension.y = 512;

	SECTION("Initialise simplex noise constant memory") {
		REQUIRE_NOTHROW(STPSimplexNoise::initialise());
	}

	SECTION("Initialise the simplex noise generator") {
		REQUIRE_NOTHROW([&sim_set]() -> void {
			sim_set.Seed = random(0, 10000).get();
			const STPSimplexNoise perm(&sim_set);
		}());
	}

	SECTION("Randomness of simplex 2D noise, different seeds map to different output") {
		float* results = nullptr;
		STPSimplexNoise* sim_d = nullptr;
		cudaMallocManaged(&results, sizeof(float) * 2, cudaMemAttachGlobal);
		cudaMalloc(&sim_d, sizeof(STPSimplexNoise) * 2);

		for (int i = 0; i < 5; i++) {
			const unsigned long long seed1 = random(0, 10000).get();
			const unsigned long long seed2 = random(0, 10000).get();
			sim_set.Seed = seed1;
			STPSimplexNoise sim1(&sim_set);
			sim_set.Seed = seed2;
			STPSimplexNoise sim2(&sim_set);

			cudaMemcpy(sim_d, &sim1, sizeof(STPSimplexNoise), cudaMemcpyHostToDevice);
			cudaMemcpy(sim_d + 1, &sim2, sizeof(STPSimplexNoise), cudaMemcpyHostToDevice);
			float2 index;
			index.x = random(-100.0f, 100.0f).get();
			index.y = random(-100.0f, 100.0f).get();

			sim2d << <1, 1 >> > (sim_d, index, results);
			sim2d << <1, 1 >> > (sim_d + 1, index, results + 1);
			cudaDeviceSynchronize();

			if (seed1 != seed2) {
				INFO("Simplex noise has a chance of getting the same value even with seeds are not equal, better run it again.");
				REQUIRE(results[0] != results[1]);
			}
			else {
				//We don't care about determinisisity now
				WARN("Same seed has been generated, skip this pass");
				SUCCEED();
			}
		}

		cudaFree(results);
		cudaFree(sim_d);
	}

	SECTION("Deterministisity test, the same input always maps to the same output") {
		float* result = nullptr;
		STPSimplexNoise* sim_d = nullptr;
		const unsigned long long seed = random(0, 10000).get();
		sim_set.Seed = seed;
		const STPSimplexNoise sim(&sim_set);
		cudaMallocManaged(&result, sizeof(int) * 5, cudaMemAttachGlobal);
		cudaMalloc(&sim_d, sizeof(STPSimplexNoise));
		cudaMemcpy(sim_d, &sim, sizeof(STPSimplexNoise), cudaMemcpyHostToDevice);

		float prev[5];
		float2 index[5];
		for (int i = 0; i < 5; i++) {
			index[i].x = random(-1000.0f, 1000.0f).get();
			index[i].y = random(-1000.0f, 1000.0f).get();
		}
		for (int i = 0; i < 5; i++) {
			sim2d << <1, 1 >> > (sim_d, index[i], result + i);
			cudaDeviceSynchronize();
			prev[i] = result[i];
		}
		for (int i = 0; i < 5; i++) {
			sim2d << <1, 1 >> > (sim_d, index[i], result + i);
			cudaDeviceSynchronize();
			INFO("The value was " << result[i] << " at index " << i << ", but got " << prev[i]);
			REQUIRE(result[i] == prev[i]);
		}

		cudaFree(result);
		cudaFree(sim_d);
	}
}