//Catch2
#include <catch2/catch_test_macros.hpp>
//Generators
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
//Matcher
#include <catch2/matchers/catch_matchers_string.hpp>

//SuperTerrain+/GPGPU
#include <SuperTerrain+/GPGPU/STPRuntimeCompilable.h>

//Utils
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceMemory.h>
#include <SuperTerrain+/Utility/STPFile.h>

#include <SuperTerrain+/Exception/STPCompilationError.h>
#include <SuperTerrain+/Exception/STPSerialisationError.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>

//CUDA
#include <cuda_runtime.h>

//GLM
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

using namespace SuperTerrainPlus;

using glm::uvec2;
using glm::vec2;
using glm::vec4;
using glm::mat4;
using glm::value_ptr;

using std::string;
using std::cout;
using std::cerr;
using std::endl;

class RTCTester : protected STPRuntimeCompilable {
private:

	constexpr static char SourceLocation[] = "./TestData/MatrixArithmetic.rtc";

	constexpr static uvec2 MatDim = uvec2(4u);

	//We need to guarantee the life-time of those string
	constexpr static char TransformAdd[] = "transform<MatrixOperator::Addition>";
	constexpr static char TransformSub[] = "transform<MatrixOperator::Subtraction>";

protected:

	constexpr static char SourceName[] = "MatrixArithmetic";

	constexpr static char ExternalHeaderSource[] = 
		"#pragma once								\n"
		"#ifndef _MatrixArithmeticVersion_			\n"
		"#define _MatrixArithmeticVersion_			\n"
		"#define MATRIX_ARITHETIC_VERSION \"6.6.6\"	\n"
		"#endif//_MatrixArithmeticVersion_			\n";
	constexpr static char ExternalHeaderName[] = "MatrixArithmeticVersion";

	STPSmartDeviceMemory::STPDeviceMemory<float[]> MatA, MatB, MatOut, MatBuffer;

	CUfunction MattransformAdd, MattransformSub, Matscale;

	void testCompilation(bool test_enable, bool attach_header) {
		using namespace std::string_literals;
		const string Capability = "-arch=sm_"s + std::to_string(RTCTester::getArchitecture(0));
		//settings
		STPSourceInformation src_info;
		src_info.Option
			["-std=c++17"]
			[Capability.c_str()]
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
			[RTCTester::TransformAdd]
			[RTCTester::TransformSub]
			["scale"];
		if (attach_header) {
			//define a macro to enable external header testing
			src_info.Option
				["-DSTP_TEST_EXTERNAL_HEADER"];
			//attach header source code to the database
			REQUIRE(this->attachHeader(RTCTester::ExternalHeaderName, RTCTester::ExternalHeaderSource));
			//add to compiler list
			src_info.ExternalHeader
				[RTCTester::ExternalHeaderName];
		}

		//read source code
		const STPFile src = STPFile(RTCTester::SourceLocation);

		//compile
		auto startCompile = [&src_info, &src, this]() {
			string log;
			try {
				log = this->compileSource(RTCTester::SourceName, *src, src_info);
			}
			catch (const STPException::STPCompilationError& ce) {
				//compile time error
				cerr << ce.what() << endl;
				//rethrow it
				throw;
			}
			//print the log (if any)
			if (!log.empty()) {
				cout << log << endl;
			}
		};
		CHECKED_IF(test_enable) {
			CHECKED_IF(attach_header) {
				REQUIRE_THROWS_WITH(startCompile(), Catch::Matchers::ContainsSubstring("6.6.6") && Catch::Matchers::ContainsSubstring("PASS"));
			}
			CHECKED_ELSE(attach_header) {
				REQUIRE_NOTHROW(startCompile());
			}
		}
		CHECKED_ELSE(test_enable) {
			REQUIRE_THROWS_WITH(startCompile(), Catch::Matchers::ContainsSubstring("STP_TEST_ENABLE"));
		}
	}

	void testLink() {
		//log
		constexpr static unsigned int logSize = 1024u;
		char linker_info[logSize], linker_error[logSize];
		char module_info[logSize], module_error[logSize];

		STPLinkerInformation link_info;
		link_info.LinkerOption
		(CU_JIT_OPTIMIZATION_LEVEL, (void*)0u)
			(CU_JIT_LOG_VERBOSE, (void*)1)
			(CU_JIT_INFO_LOG_BUFFER, linker_info)
			(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, (void*)(uintptr_t)logSize)
			(CU_JIT_ERROR_LOG_BUFFER, linker_error)
			(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, (void*)(uintptr_t)logSize);
		link_info.getDataOption(RTCTester::SourceName)
			(CU_JIT_MAX_REGISTERS, (void*)72u);
		link_info.ModuleOption
		(CU_JIT_INFO_LOG_BUFFER, module_info)
			(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, (void*)(uintptr_t)logSize)
			(CU_JIT_ERROR_LOG_BUFFER, module_error)
			(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, (void*)(uintptr_t)logSize)
			(CU_JIT_LOG_VERBOSE, (void*)1);

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
		STPcudaCheckErr(cuMemcpyHtoD(matrixDim_d, &matrixDim_h, matrixDimSize));

		//get function pointer
		STPcudaCheckErr(cuModuleGetFunction(&this->MattransformAdd, program, name.at(RTCTester::TransformAdd)));
		STPcudaCheckErr(cuModuleGetFunction(&this->MattransformSub, program, name.at(RTCTester::TransformSub)));
		STPcudaCheckErr(cuModuleGetFunction(&this->Matscale, program, name.at("scale")));
	}

	mat4 matrixTransform(CUfunction func, const mat4& matA, const mat4& matB, float factor) {
		assert(func != this->Matscale);

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
		STPcudaCheckErr(cuStreamSynchronize(0));

		//copy the result back
		mat4 matOut;
		STPcudaCheckErr(cudaMemcpy(value_ptr(matOut), this->MatOut.get(), sizeof(mat4), cudaMemcpyDeviceToHost));

		return matOut;
	}

public:

	RTCTester() : STPRuntimeCompilable() {
		//context has been init at the start of the test program
		const unsigned int matSize = RTCTester::MatDim.x * RTCTester::MatDim.y;
		this->MatA = STPSmartDeviceMemory::makeDevice<float[]>(matSize);
		this->MatB = STPSmartDeviceMemory::makeDevice<float[]>(matSize);
		this->MatOut = STPSmartDeviceMemory::makeDevice<float[]>(matSize);
		this->MatBuffer = STPSmartDeviceMemory::makeDevice<float[]>(matSize);
	}

};

static constexpr char Nonsense[] = "Blah.blah";

SCENARIO_METHOD(RTCTester, "STPRuntimeCompilable manages runtime CUDA scripts and runs the kernel", "[GPGPU][STPRuntimeCompilable]") {

	GIVEN("A RTC version of diversity generator with custom implementation and runtime script") {

		WHEN("The source code contains error") {

			THEN("Error should be thrown out to notify the user with compiler log") {
				//set false to not define a macro to make it throws an intended error
				this->testCompilation(false, false);
			}

		}

		WHEN("Retrieve source lowered name from a non-existing source code") {

			THEN("No lowered name is returned with an error thrown") {
				REQUIRE_THROWS_AS(this->retrieveSourceLoweredName(Nonsense), STPException::STPMemoryError);
			}

		}

		WHEN("A piece of working source code is added to the program") {

			THEN("Program can be compiled without errors") {
				this->testCompilation(true, false);

				AND_THEN("After linking, data can be sent to kernel, after the execution result can be retrieved, and correctness is verified") {
					this->testLink();
					REQUIRE_NOTHROW(this->prepData());
					//round the number to 1 d.p. to avoid float rounding issue during assertion
					//compilation is a slow process, so we only test it once
					const auto Data = GENERATE(take(1, chunk(18, map<float>([](auto f) { return roundf(f * 10.0f) / 10.0f; }, random(-6666.0f, 6666.0f)))));
					//kernel execution for matrix addition
					const mat4 matA = glm::make_mat4(Data.data()), 
						matB = glm::identity<mat4>() * Data[16];
					const float scale = Data[17];
					mat4 matResult;
					REQUIRE_NOTHROW([this, &matResult, &matA, &matB, &scale]() { matResult = this->matrixTransform(this->MattransformAdd, matA, matB, scale); }());
					REQUIRE(matResult == (matA + matB) * scale);

					//again for subtraction
					REQUIRE_NOTHROW([this, &matResult, &matA, &matB, &scale]() { matResult = this->matrixTransform(this->MattransformSub, matA, matB, scale); }());
					REQUIRE(matResult == (matA - matB) * scale);

				}

				AND_WHEN("The same piece of source code is re-added to the program") {

					THEN("Diversity generator does not allow the same source code to be added") {
						STPSourceInformation test_info;
						const string randomSource("#error This source code is not intended to be compiled\n");
						REQUIRE_THROWS_AS(this->compileSource(RTCTester::SourceName, randomSource, test_info), STPException::STPMemoryError);
					}

				}

			}

			AND_WHEN("Source is asked to be discarded") {

				THEN("Source code is discarded if it has been compiled before") {
					this->testCompilation(true, false);
					REQUIRE(this->discardSource(RTCTester::SourceName));
				}

				THEN("Nothing should happen if source code name is not found in the database") {
					REQUIRE_FALSE(this->discardSource(Nonsense));
				}

			}

			AND_GIVEN("An external header") {

				WHEN("Header is attached to the program database") {

					THEN("Header is recognised by the compiler and can be used alongside with the source file") {
						this->testCompilation(true, true);

						AND_WHEN("The same header is attached to the program") {

							THEN("Program should not allow header with the same name to be added") {
								REQUIRE_FALSE(this->attachHeader(RTCTester::ExternalHeaderName, RTCTester::ExternalHeaderSource));
							}

						}

					}

					AND_WHEN("Header is asked to be discarded") {

						THEN("Header should be discarded if it exists") {
							REQUIRE(this->attachHeader(Nonsense, Nonsense));
							REQUIRE(this->detachHeader(Nonsense));
						}

						THEN("Nothing should happen if header name is not found") {
							REQUIRE_FALSE(this->detachHeader(Nonsense));
						}

					}

				}

			}

			AND_GIVEN("An archive file") {

				WHEN("Archive is attached to the program database") {
					//TODO: I don't have an archive on my hand to be tested

					AND_WHEN("Archive is asked to be discarded") {

						THEN("Archive should be discarded if it exists") {
							REQUIRE(this->attachArchive(Nonsense, Nonsense));
							REQUIRE(this->detachArchive(Nonsense));
						}

						THEN("Nothing should happen if the archive name is not found") {
							REQUIRE_FALSE(this->detachArchive(Nonsense));
						}

					}

				}

			}

		}

	}

}