#pragma once

//Catch2
#include <catch2/catch_test_macros.hpp>
//Generators
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

//Proto Code
#include "Compute/SobelFilter.cuh"

//SuperTerrain+/Utility
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Utility/STPSmartDeviceMemory.h>

//GLM
#include <glm/vec2.hpp>

#include <limits>

using namespace SuperTerrainPlus;

using glm::uvec2;
using glm::vec2;

class PrototypeConvolution {
protected:

	constexpr static uvec2 ImageDimension = uvec2(32u);
	constexpr static unsigned int ImagePixelCount = ImageDimension.x * ImageDimension.y;

private:

	inline static bool init = false;

	inline void fillHostData() {
		//round to 2 d.p.
		const auto data = GENERATE(take(1, chunk(PrototypeConvolution::ImagePixelCount, map<float>([](auto f) { return roundf(f * 100.0f) / 100.0f; }, random(-1.0f, 1.0f)))));
		for (unsigned int i = 0; i < PrototypeConvolution::ImagePixelCount; i++) {
			this->ImageHostIn[i] = data[i];
		}
	}

protected:

	float ImageHostIn[ImagePixelCount];
	STPSmartDeviceMemory::STPDeviceMemory<float[]> ImageDeviceIn, ImageDeviceOut;

	template<SobelFilterMode M>
	void launchKernel(float* hostOutput) {
		//get kernel parameters
		float* output = this->ImageDeviceOut.get(),
			*input = this->ImageDeviceIn.get();
		void* args[] = {
			&output,
			&input
		};

		//launch
		dim3 blockDim = dim3(8u, 8u),
			gridDim = dim3(4u, 4u);
		size_t cacheSize = 0ull;
		if constexpr (M != SobelFilterMode::VanillaConvolution) {
			cacheSize = (blockDim.x + 2u) * (blockDim.y + 2u) * sizeof(float);
		}
		STPcudaCheckErr(cudaLaunchKernel(&sobelFilter<M>, gridDim, blockDim, args, cacheSize, 0));
		STPcudaCheckErr(cudaGetLastError());
		STPcudaCheckErr(cudaStreamSynchronize(0));

		//retrieve data
		STPcudaCheckErr(cudaMemcpy(hostOutput, this->ImageDeviceOut.get(), sizeof(float) * PrototypeConvolution::ImagePixelCount, cudaMemcpyDeviceToHost));
	}

	void compareImage(float* reference, float* target) const {
		const auto index = GENERATE(take(2, chunk(12, random(0u, PrototypeConvolution::ImagePixelCount - 1u))));
		for (const auto i : index) {
			CHECK(fabs(target[i] - reference[i]) < 10.0f * std::numeric_limits<float>::epsilon());
		}
	}

public:

	PrototypeConvolution() : 
		ImageDeviceIn(STPSmartDeviceMemory::makeDevice<float[]>(PrototypeConvolution::ImagePixelCount)), 
		ImageDeviceOut(STPSmartDeviceMemory::makeDevice<float[]>(PrototypeConvolution::ImagePixelCount)) {
		//fill in data for testing
		this->fillHostData();
		//copy to device
		STPcudaCheckErr(cudaMemcpy(this->ImageDeviceIn.get(), this->ImageHostIn, sizeof(float) * PrototypeConvolution::ImagePixelCount, cudaMemcpyHostToDevice));

		//init device data
		if (!PrototypeConvolution::init) {
			//initialise constant memory
			STPcudaCheckErr(cudaMemcpyToSymbol(getDimensionSymbol(), &PrototypeConvolution::ImageDimension, sizeof(uvec2), 0ull, cudaMemcpyHostToDevice));

			PrototypeConvolution::init = true;
		}
	}

};

SCENARIO_METHOD(PrototypeConvolution, "STPHeightfieldGenerator generates rendering buffer using sobel filtering and outputs the same result before and after optimisation", 
	"[Prototype][GPGPU][STPHeightfieldGenerator]") {

	GIVEN("A various of implementations of convolution kernel") {

		AND_GIVEN("A result from launching a unoptimised version of kernel") {
			//naive implementation with zero optimisation
			float ReferenceOutput[PrototypeConvolution::ImagePixelCount];
			this->launchKernel<SobelFilterMode::VanillaConvolution>(ReferenceOutput);

			WHEN("Launch a kernel with shared memory optimisation") {
				//optimised using shared memory
				float RemappedSharedOutput[PrototypeConvolution::ImagePixelCount];
				float CoalescedSharedOutput[PrototypeConvolution::ImagePixelCount];

				this->launchKernel<SobelFilterMode::RemappedCacheLoadConvolution>(RemappedSharedOutput);
				this->launchKernel<SobelFilterMode::CoalescedCacheLoadConvolution>(CoalescedSharedOutput);

				THEN("The result of an optimised kernel should be the same as the unoptimised version") {
					this->compareImage(ReferenceOutput, RemappedSharedOutput);
					this->compareImage(ReferenceOutput, CoalescedSharedOutput);
				}

			}

		}

	}
}