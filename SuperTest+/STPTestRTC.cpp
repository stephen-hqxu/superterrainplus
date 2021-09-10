#pragma once

//Catch2
#include <catch2/catch_test_macros.hpp>
//Generators
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
//Matcher
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

//SuperTerrain+/GPGPU
#include <SuperTerrain+/GPGPU/STPDiversityGeneratorRTC.h>

//Utils
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Utility/STPSmartDeviceMemory.h>
#include <SuperTerrain+/Utility/STPSmartDeviceMemory.tpp>

#include <SuperTerrain+/Utility/Exception/STPCompilationError.h>

//GLM
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

//CUDA
#include <cuda_runtime.h>
//RTC header testing
#include "./TestData/MatrixArithmetic.rtch"

#include <iostream>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPCompute;

using glm::uvec2;
using glm::vec2;
using glm::vec4;
using glm::mat4;
using glm::value_ptr;

using std::string;
using std::cout;
using std::cerr;
using std::endl;

class RTCTester : protected STPDiversityGeneratorRTC {
private:

	constexpr static char SourceLocation[] = "./TestData/MatrixArithmetic.rtc";
	constexpr static char SourceName[] = "MatrixArithmetic";

	constexpr static uvec2 MatDim = uvec2(4u);

	//We need to guarantee the life-time of those string
	inline const static string TransformAdd = "transform<" + string("MatrixOperator::Addition") + ">";
	inline const static string TransformSub = "transform<" + string("MatrixOperator::Subtraction") + ">";

protected:

	STPSmartDeviceMemory::STPDeviceMemory<float[]> MatA, MatB, MatOut, MatBuffer;

	CUfunction MattransformAdd, MattransformSub, Matscale;

	void testCompilation(bool test_enable) {
		//settings
		STPSourceInformation src_info;
		src_info.Option
			["-std=c++17"]
			["-arch=compute_75"]
			["-fmad=false"];
		if (test_enable) {
			//it's a define switch to test if compiler options are taken by the compiler
			//if this symbol is not defined it should throw an error
			src_info.Option["-DSTP_TEST_ENABLE"];
		}
		src_info.NameExpression
			//constant variable
			["MatrixDimension"]
			//global functions
			[RTCTester::TransformAdd.c_str()]
			[RTCTester::TransformSub.c_str()]
			["scale"];

		//read source code
		string src;
		REQUIRE_NOTHROW([&src, this]() { src = this->readSource(RTCTester::SourceLocation); }());

		//compile
		auto startCompile = [&src_info, &src, this]() {
			string log;
			log = this->compileSource(RTCTester::SourceName, src, src_info);
			//print the log (if any)
			if (!log.empty()) {
				cout << log << endl;
			}
		};
		CHECKED_IF(test_enable) {
			REQUIRE_NOTHROW(startCompile());
		}
		CHECKED_ELSE(test_enable) {
			REQUIRE_THROWS_WITH(startCompile(), Catch::Matchers::Contains("STP_TEST_ENABLE"));
		}
	}

	void testLink() {
		//log
		constexpr static unsigned int logSize = 1024u;
		char linker_info[logSize], linker_error[logSize];
		char module_info[logSize], module_error[logSize];

		STPLinkerInformation link_info;
		link_info.LinkerOption
		(CU_JIT_OPTIMIZATION_LEVEL, (void*)1u)
			(CU_JIT_LOG_VERBOSE, (void*)1)
			(CU_JIT_INFO_LOG_BUFFER, linker_info)
			(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, (void*)logSize)
			(CU_JIT_ERROR_LOG_BUFFER, linker_error)
			(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, (void*)logSize);
		link_info.ModuleOption
		(CU_JIT_INFO_LOG_BUFFER, module_info)
			(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, (void*)logSize)
			(CU_JIT_ERROR_LOG_BUFFER, module_error)
			(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, (void*)logSize);

		//link
		REQUIRE_NOTHROW([&]() {
			try {
				this->linkProgram(link_info, CU_JIT_INPUT_PTX);
			}
			catch (...) {
				//print error log
				cerr << linker_error << endl;
				cerr << module_error << endl;
				throw;
			}
			//print info log
			cout << linker_info << endl;
			cout << module_info << endl;
		}());
	}

	void prepData() {
		const STPLoweredName& name = this->retrieveSourceLoweredName(RTCTester::SourceName);
		auto program = this->getGeneratorModule();
		CUdeviceptr matrixDim_d;
		size_t matrixDimSize;

		//get the pointer to the variable
		const uint2 matrixDim_h = make_uint2(RTCTester::MatDim.x, RTCTester::MatDim.y);
		STPcudaCheckErr(cuModuleGetGlobal(&matrixDim_d, &matrixDimSize, program, name.at("MatrixDimension")));
		STPcudaCheckErr(cuMemcpyHtoD(matrixDim_d, &matrixDim_h, sizeof(uint2)));

		//get function pointer
		STPcudaCheckErr(cuModuleGetFunction(&this->MattransformAdd, program, name.at(RTCTester::TransformAdd)));
		STPcudaCheckErr(cuModuleGetFunction(&this->MattransformSub, program, name.at(RTCTester::TransformSub)));
		STPcudaCheckErr(cuModuleGetFunction(&this->Matscale, program, name.at("scale")));
	}

	mat4 matrixTransform(CUfunction func, const mat4& matA, const mat4& matB, float factor) {
		assert(func != this->Matscale);
		auto program = this->getGeneratorModule();

		//copy input to device
		STPcudaCheckErr(cudaMemcpy(this->MatA.get(), value_ptr(matA), sizeof(mat4), cudaMemcpyHostToDevice));
		STPcudaCheckErr(cudaMemcpy(this->MatB.get(), value_ptr(matB), sizeof(mat4), cudaMemcpyHostToDevice));

		//launch kernel
		{
			//transform
			float* output = this->MatBuffer.get(),
				*ma = this->MatA.get(),
				*mb = this->MatB.get();
			void* args[] = {
				&output,
				&ma,
				&mb
			};
			STPcudaCheckErr(cuLaunchKernel(func,
				1u, 1u, 1u,
				8u, 4u, 1u,
				0u, 0, args, nullptr
			));
			STPcudaCheckErr(cudaGetLastError());
		}
		{
			//scale
			float* output = this->MatOut.get(),
				*input = this->MatBuffer.get();
			void* args[] = {
				&output,
				&input,
				&factor
			};
			STPcudaCheckErr(cuLaunchKernel(this->Matscale,
				1u, 1u, 1u,
				8u, 4u, 1u,
				0u, 0, args, nullptr
			));
		}
		STPcudaCheckErr(cuCtxSynchronize());

		//copy the result back
		mat4 matOut;
		STPcudaCheckErr(cudaMemcpy(value_ptr(matOut), this->MatOut.get(), sizeof(mat4), cudaMemcpyDeviceToHost));

		return matOut;
	}

public:

	RTCTester() : STPDiversityGeneratorRTC() {
		//context has been init at the start of the test program
		const unsigned int matSize = RTCTester::MatDim.x * RTCTester::MatDim.y;
		this->MatA = STPSmartDeviceMemory::makeDevice<float[]>(matSize);
		this->MatB = STPSmartDeviceMemory::makeDevice<float[]>(matSize);
		this->MatOut = STPSmartDeviceMemory::makeDevice<float[]>(matSize);
		this->MatBuffer = STPSmartDeviceMemory::makeDevice<float[]>(matSize);
	}

	void operator()(STPFreeSlipFloatTextureBuffer& heightmap, const STPFreeSlipGenerator::STPFreeSlipSampleManagerAdaptor& biomemap, vec2 offset, cudaStream_t stream) const override {
		WARN(__FUNCTION__ << "is not supposed to be called");
	}

};

SCENARIO_METHOD(RTCTester, "STPDiversityGeneratorRTC manages runtime CUDA scripts and runs the kernel", "[GPGPU][STPDiversityGeneratorRTC]") {

	GIVEN("A RTC version of diversity generator with custom implementation and runtime script") {

		WHEN("The source code contains error") {

			THEN("Error should be thrown out to notify the user with compiler log") {
				//set false to not define a macro to make it throws an intended error
				this->testCompilation(false);
			}

		}

		WHEN("The source code works fine") {

			THEN("Program can be compiled and linked without errors") {
				this->testCompilation(true);
				this->testLink();
				REQUIRE_NOTHROW(this->prepData());

				AND_THEN("Data can be sent to kernel, after execution result can be retrieved, and correctness is verified") {
					//round the number to 1 d.p. to avoid float rounding issue during assertion
					const auto Data = GENERATE(take(2, chunk(18, map<float>([](auto f) { return roundf(f * 10.0f) / 10.0f; }, random(-6666.0f, 6666.0f)))));
					//kernel exeuction for matrix addition
					const mat4 matA = mat4(
						Data[0], Data[1], Data[2], Data[3],
						Data[4], Data[5], Data[6], Data[7],
						Data[8], Data[9], Data[10], Data[11],
						Data[12], Data[13], Data[14], Data[15]
					), matB = glm::identity<mat4>() * Data[16];
					const float scale = Data[17];
					mat4 matResult;
					REQUIRE_NOTHROW([this, &matResult, &matA, &matB, &scale]() { matResult = this->matrixTransform(this->MattransformAdd, matA, matB, scale); }());
					REQUIRE(matResult == (matA + matB) * scale);

					//again for subtraction
					REQUIRE_NOTHROW([this, &matResult, &matA, &matB, &scale]() { matResult = this->matrixTransform(this->MattransformSub, matA, matB, scale); }());
					REQUIRE(matResult == (matA - matB) * scale);

				}

			}

		}

	}

}