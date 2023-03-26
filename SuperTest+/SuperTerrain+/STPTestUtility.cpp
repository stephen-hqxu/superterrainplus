//Catch2
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
//Matcher
#include <catch2/matchers/catch_matchers_string.hpp>

//CUDA
#include "../STPTestInformation.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>

//SuperTerrain+/SuperTerrain+/Utility
#include <SuperTerrain+/Utility/STPThreadPool.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>
#include <SuperTerrain+/Utility/Memory/STPObjectPool.h>
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceMemory.h>
//SuperTerrain+/SuperTerrain+/Exception
#include <SuperTerrain+/Exception/API/STPCUDAError.h>
#include <SuperTerrain+/Exception/STPNumericDomainError.h>

#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <optional>
#include <utility>

using Catch::Matchers::ContainsSubstring;

using namespace SuperTerrainPlus;

using std::atomic;
using std::optional;
using std::unique_ptr;
using std::make_unique;
using std::future;

class ThreadPoolTester {
private:

	static void sleep() {
		std::this_thread::sleep_for(std::chrono::milliseconds(10ull));
	}

protected:

	static constexpr size_t WorkBatchSize = 4u;

	//use optional so we can control the life-time of the thread pool for easy testing
	optional<STPThreadPool> Pool;

	static unsigned int busyWork(const unsigned int value) {
		ThreadPoolTester::sleep();
		return value;
	}

	static void busyWorkNoReturn(atomic<bool>& flag) {
		ThreadPoolTester::sleep();
		flag = !flag;
	}

	[[noreturn]] static void brokenWork(const unsigned int value) {
		ThreadPoolTester::busyWork(value);
		throw std::runtime_error("This test case is intended to fail");
	}

	void submitManyWorks(atomic<unsigned int>& counter) {
		for (size_t work = 0u; work < WorkBatchSize; work++) {
			this->Pool->enqueueDetached([&counter](unsigned int value) {
				ThreadPoolTester::busyWork(value);
				counter++;
			}, static_cast<unsigned int>(work));
		}
	}

	//return the actual number of block sent
	template<size_t NB = 2u>
	unsigned int submitLoopWork(atomic<unsigned int>& counter, const unsigned int start, const unsigned int finish) {
		const auto accumulateCount = [&counter](auto, const auto begin, const auto end) -> void {
			counter += ThreadPoolTester::busyWork(end - begin);
		};
		//submit
		auto LoopFuture = this->Pool->enqueueLoop<NB>(accumulateCount, start, finish);
		//wait
		return std::visit([](auto& result) {
			for (auto& f : result) {
				f.get();
			}
			return static_cast<unsigned int>(result.size());
		}, LoopFuture);
	}

public:

	ThreadPoolTester() : Pool(1u) {

	}

};

SCENARIO_METHOD(ThreadPoolTester, "STPThreadPool used in a multi-threaded workload", "[Utility][STPThreadPool]") {

	GIVEN("An invalid thread pool with zero worker") {

		THEN("Thread pool should throw an error") {
			REQUIRE_THROWS_AS(STPThreadPool(0u), STPException::STPNumericDomainError);
		}

	}

	GIVEN("A valid thread pool object") {

		WHEN("Work can be done without generating any error") {

			AND_WHEN("The work returns a value") {
				const unsigned int Work = GENERATE(take(3u, random(0u, 6666u)));
				future<unsigned int> Result;
				REQUIRE_NOTHROW([&pool = *this->Pool, &Result, Work]() {
					Result = pool.enqueue(&ThreadPoolTester::busyWork, Work);
				}());

				THEN("Work is done successfully and the result can be retrieved") {
					REQUIRE(Result.get() == Work);
				}

			}

			AND_WHEN("The work does not return a value") {
				atomic<bool> Flag = false;
				future<void> Result;
				REQUIRE_NOTHROW([&pool = *this->Pool, &Result, &Flag]() {
					Result = pool.enqueue(&ThreadPoolTester::busyWorkNoReturn, std::ref(Flag));
				}());

				THEN("Work is done successfully even if it does not return any value") {
					Result.get();
					REQUIRE(Flag);
				}
			}

			AND_WHEN("The work is detached from the main thread") {
				atomic<bool> Flag = false;
				REQUIRE_NOTHROW([&pool = *this->Pool, &Flag]() {
					pool.enqueueDetached(&ThreadPoolTester::busyWorkNoReturn, std::ref(Flag));
				}());

				THEN("Work is done successfully with explicit thread pool synchronisation") {
					this->Pool->waitAll();
					REQUIRE(Flag);
				}

			}

			AND_GIVEN("Some works that require running a loop of iterations to be parallelised into equal-sized blocks") {
				atomic<unsigned int> Counter = 0u;

				WHEN("The end index is less than the begin index, resulting in negative number of iteration") {

					THEN("Loop parallelisation fails with an error") {
						REQUIRE_THROWS_AS(this->submitLoopWork(Counter, 321u, 123u), STPException::STPNumericDomainError);
					}

				}

				WHEN("The number of block is less than the number of iteration") {

					THEN("Iteration is split evenly into blocks and run in parallel; the total number of iteration is the same") {
						//deliberately using an iteration that is not divisible by the block count
						REQUIRE(this->submitLoopWork<4u>(Counter, 0u, 57u) == 4u);
						REQUIRE(Counter == 57u);
					}

				}

				WHEN("The number of block is greater than the number of iteration") {

					THEN("Only one block is launched; this block handles the entire iteration") {
						REQUIRE(this->submitLoopWork<4u>(Counter, 0u, 3u) == 1u);
						REQUIRE(Counter == 3u);
					}

				}

			}

		}

		WHEN("The work to be done throws an exception") {

			THEN("The thread pool captures the exception successfully without crashing itself") {
				future<void> Result;
				REQUIRE_NOTHROW([&pool = *this->Pool, &Result]() {
					Result = pool.enqueue(&ThreadPoolTester::brokenWork, 12345u);
				}());

				REQUIRE_THROWS_WITH(Result.get(), ContainsSubstring("intended to fail"));
			}

		}

		WHEN("The thread pool is destroyed while working") {

			THEN("The thread pool should be destroyed only when all tasks are done") {
				atomic<unsigned int> Counter = 0u;
				this->submitManyWorks(Counter);

				//kill the pool while the tasks are being done
				this->Pool.reset();
				//the destructor of the pool should not return until everything has been done
				REQUIRE(Counter == ThreadPoolTester::WorkBatchSize);
			}

		}
	}
}

SCENARIO("STPDeviceErrorHandler reports CUDA API error", "[Utility][STPDeviceErrorHandler]") {

	GIVEN("A workflow of CUDA API calls") {

		WHEN("Calling some CUDA APIs") {

			AND_WHEN("API call is successful") {

				THEN("No error is thrown") {
					CHECK_NOTHROW(STP_CHECK_CUDA(cudaError::cudaSuccess));
					CHECK_NOTHROW(STP_CHECK_CUDA(CUresult::CUDA_SUCCESS));
					CHECK_NOTHROW(STP_CHECK_CUDA(nvrtcResult::NVRTC_SUCCESS));
				}

			}

			AND_WHEN("API call throws error") {

				THEN("Error should be reported") {
					CHECK_THROWS_AS(STP_CHECK_CUDA(cudaError::cudaErrorInvalidValue), STPException::STPCUDAError);
					CHECK_THROWS_AS(STP_CHECK_CUDA(CUresult::CUDA_ERROR_INVALID_VALUE), STPException::STPCUDAError);
					CHECK_THROWS_AS(STP_CHECK_CUDA(nvrtcResult::NVRTC_ERROR_OUT_OF_MEMORY), STPException::STPCUDAError);
				}

			}

		}

	}

}

template<typename T>
struct ObjectCreator {
private:

	T Counter = static_cast<T>(123);

public:

	unique_ptr<T> operator()() {
		return make_unique<T>(this->Counter++);
	}

};

template<typename T>
using ObjectPoolTester = STPObjectPool<unique_ptr<T>, ObjectCreator<T>>;

TEMPLATE_TEST_CASE_METHOD(ObjectPoolTester, "STPObjectPool reuses object whenever possible", "[Utility][STPObjectPool]",
	float, unsigned int){
	using std::move;

	GIVEN("A fresh object pool") {

		WHEN("An object is requested") {
			auto Obj = this->request();

			THEN("The requested object should be constructed appropriately") {
				REQUIRE(*Obj == static_cast<TestType>(123));
			}

			AND_WHEN("The object is returned and re-requested") {
				this->release(move(Obj));
				auto ReObj = this->request();

				THEN("The requested object should be the same as the first object") {
					REQUIRE(*ReObj == static_cast<TestType>(123));
				}

				AND_WHEN("Another object is requested while the pool is empty") {
					auto AnotherObj = this->request();

					THEN("The newly requested object should be newly created") {
						REQUIRE(*AnotherObj == static_cast<TestType>(124));
					}

				}

			}

		}

	}

}

#define VERIFY_DATA() REQUIRE(std::all_of(ReturnedData.get(), ReturnedData.get() + 8, compareData))

SCENARIO("STPSmartDeviceMemory allocates and auto-delete device pointer", "[Utility][STPSmartDeviceMemory]") {
	
	GIVEN("A piece of host data that is needed to be copied to device memory") {
		const unique_ptr<unsigned int[]> HostData = make_unique<unsigned int[]>(8);
		const unique_ptr<unsigned int[]> ReturnedData = make_unique<unsigned int[]>(8);

		const unsigned int Data = GENERATE(take(3, random(0u, 66666666u)));
		std::fill_n(HostData.get(), 8, Data);

		WHEN("A smart device memory is requested") {
			constexpr static size_t RowSize2D = sizeof(unsigned int) * 4u;

			auto DeviceData = STPSmartDeviceMemory::makeDevice<unsigned int[]>(8);
			auto StreamedDeviceData = STPSmartDeviceMemory::makeStreamedDevice<unsigned int[]>(STPTestInformation::TestDeviceMemoryPool, 0, 8);
			auto PitchedDeviceData = STPSmartDeviceMemory::makePitchedDevice<unsigned int[]>(4, 2);

			THEN("Smart device memory can be used like normal memory") {
				constexpr static size_t DataSize = sizeof(unsigned int) * 8u;
				const auto compareData = [Data](const auto i) {
					return i == Data;
				};

				//copy back and forth
				//regular device memory
				STP_CHECK_CUDA(cudaMemcpy(DeviceData.get(), HostData.get(), DataSize, cudaMemcpyHostToDevice));
				STP_CHECK_CUDA(cudaMemcpy(ReturnedData.get(), DeviceData.get(), DataSize, cudaMemcpyDeviceToHost));
				VERIFY_DATA();

				//stream-ordered device memory
				STP_CHECK_CUDA(cudaMemcpyAsync(StreamedDeviceData.get(), HostData.get(), DataSize, cudaMemcpyHostToDevice, 0));
				STP_CHECK_CUDA(cudaMemcpyAsync(ReturnedData.get(), StreamedDeviceData.get(), DataSize, cudaMemcpyDeviceToHost, 0));
				STP_CHECK_CUDA(cudaStreamSynchronize(0));
				VERIFY_DATA();

				//pitched device memory
				STP_CHECK_CUDA(cudaMemcpy2D(PitchedDeviceData.get(), PitchedDeviceData.Pitch, HostData.get(),
					RowSize2D, RowSize2D, 2, cudaMemcpyHostToDevice));
				STP_CHECK_CUDA(cudaMemcpy2D(ReturnedData.get(), RowSize2D, PitchedDeviceData.get(),
					PitchedDeviceData.Pitch, RowSize2D, 2, cudaMemcpyDeviceToHost));
				VERIFY_DATA();
			}

		}

	}

}