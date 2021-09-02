#pragma once

//Catch2
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

//SuperTerrain+/Utility
#include <Utility/STPThreadPool.h>
#define STP_EXCEPTION_ON_ERROR
#define STP_SUPPRESS_ERROR_MESSAGE
#include <Utility/STPDeviceErrorHandler.h>
#include <Utility/STPSmartStream.h>
#include <Utility/STPMemoryPool.h>
//SuperTerrain+/Utility/Exception
#include <Utility/Exception/STPCUDAError.h>
#include <Utility/Exception/STPBadNumericRange.h>

//CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>

using namespace SuperTerrainPlus;

class ThreadPoolTester : protected STPThreadPool {
protected:

	static unsigned int busyWork(unsigned int value) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		return value;
	}

public:

	ThreadPoolTester() : STPThreadPool(1u) {

	}

};

SCENARIO_METHOD(ThreadPoolTester, "STPThreadPool used in a multi-threaded workload", "[Utility][STPThreadPool]") {

	GIVEN("A invalid thread pool with zero worker") {

		THEN("Thread pool should throw an error") {
			REQUIRE_THROWS_AS(STPThreadPool(0u), STPException::STPBadNumericRange);
		}

	}

	GIVEN("A valid thread pool object") {

		THEN("New thread pool should be ready to go") {
			REQUIRE(this->isRunning());
			REQUIRE(this->size() == 0ull);
		}

		WHEN("Some works needs to be done") {
			const unsigned int work = GENERATE(take(3u, random(0u, 6666u)));
			std::future<unsigned int> result;

			THEN("Thread pool should be working on the task") {
				STPThreadPool& pool = dynamic_cast<STPThreadPool&>(*this);
				REQUIRE_NOTHROW([&pool, &result, work]() {
					result = pool.enqueue_future(&ThreadPoolTester::busyWork, work);
				}());

				AND_THEN("Work can be done successfully") {
					//get the work done
					REQUIRE(result.get() == work);

					//thread pool should be idling
					REQUIRE(this->size() == 0ull);
				}
			}

		}
	}
}

SCENARIO("STPDeviceErrorHandler reports CUDA API error", "[Utility][STPDeviceErrorHandler]") {

	GIVEN("A workflow of CUDA API calls") {

		WHEN("Calling some CUDA APIs") {

			AND_WHEN("API call is successful") {
				
				THEN("No error is thrown") {
					CHECK_NOTHROW(STPcudaCheckErr(cudaError::cudaSuccess));
					CHECK_NOTHROW(STPcudaCheckErr(CUresult::CUDA_SUCCESS));
					CHECK_NOTHROW(STPcudaCheckErr(nvrtcResult::NVRTC_SUCCESS));
				}

			}

			AND_WHEN("API call throws error") {

				THEN("Error should be reported") {
					CHECK_THROWS_AS(STPcudaCheckErr(cudaError::cudaErrorInvalidValue), STPException::STPCUDAError);
					CHECK_THROWS_AS(STPcudaCheckErr(CUresult::CUDA_ERROR_INVALID_VALUE), STPException::STPCUDAError);
					CHECK_THROWS_AS(STPcudaCheckErr(nvrtcResult::NVRTC_ERROR_OUT_OF_MEMORY), STPException::STPCUDAError);
				}

			}

		}

	}

}

SCENARIO_METHOD(STPSmartStream, "STPSmartStream manages CUDA stream smartly", "[Utility][STPSmartStream]") {
	cudaStream_t stream = static_cast<cudaStream_t>(*this);

	GIVEN("A smart stream object") {

		WHEN("Some data needs to be done by CUDA") {
			const unsigned long long Original = GENERATE(take(3u, random(0ull, 1313131313ull)));

			THEN("CUDA should be doing work on the stream without throwing errors") {
				unsigned long long Destination;
				//actually, host to host copy is synchronous, but CUDA will still use the stream to make sure all previous works are done.
				REQUIRE_NOTHROW(STPcudaCheckErr(cudaMemcpyAsync(&Destination, &Original, sizeof(unsigned long long), cudaMemcpyHostToHost, stream)));
				
				AND_THEN("Work done should be verified") {
					REQUIRE_NOTHROW(STPcudaCheckErr(cudaStreamSynchronize(stream)));
					REQUIRE(Destination == Original);
				}
			}

		}

	}

}

#define REQUEST_MEMORY static_cast<unsigned long long*>(this->request(sizeof(unsigned long long)))

TEMPLATE_TEST_CASE_METHOD_SIG(STPMemoryPool, "STPMemoryPool reuses memory whenever possible", "[Utility][STPMemoryPool]", 
	((STPMemoryPoolType T), T), STPMemoryPoolType::Regular, STPMemoryPoolType::Pinned) {

	GIVEN("A fresh memory pool") {

		WHEN("Memory needs to be requested and released") {
			const unsigned long long Data = GENERATE(take(3u, random(0ull, 66666666ull)));
			unsigned long long* Mem = REQUEST_MEMORY;
			//write something
			Mem[0] = Data;
			this->release(Mem);
			
			THEN("When re-requesting memory it should be reused") {
				Mem = REQUEST_MEMORY;
				REQUIRE(Mem[0] == Data);
			}

			this->release(Mem);

		}

	}

}