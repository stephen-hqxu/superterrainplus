//Catch2
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
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
#include <SuperTerrain+/Utility/STPDeviceLaunchSetup.cuh>
#include <SuperTerrain+/Utility/Memory/STPObjectPool.h>
//SuperTerrain+/SuperTerrain+/Exception
#include <SuperTerrain+/Exception/API/STPCUDAError.h>
#include <SuperTerrain+/Exception/STPNumericDomainError.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <atomic>
#include <optional>
#include <utility>

using Catch::Matchers::ContainsSubstring;

namespace Err = SuperTerrainPlus::STPException;
namespace DevLach = SuperTerrainPlus::STPDeviceLaunchSetup;

namespace Info = STPTestInformation;

using SuperTerrainPlus::STPThreadPool, SuperTerrainPlus::STPObjectPool;

using glm::uvec2, glm::uvec3;
using glm::make_vec2, glm::make_vec3;

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
			REQUIRE_THROWS_AS(STPThreadPool(0u), Err::STPNumericDomainError);
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
						REQUIRE_THROWS_AS(this->submitLoopWork(Counter, 321u, 123u), Err::STPNumericDomainError);
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
					CHECK_THROWS_AS(STP_CHECK_CUDA(cudaError::cudaErrorInvalidValue), Err::STPCUDAError);
					CHECK_THROWS_AS(STP_CHECK_CUDA(CUresult::CUDA_ERROR_INVALID_VALUE), Err::STPCUDAError);
					CHECK_THROWS_AS(STP_CHECK_CUDA(nvrtcResult::NVRTC_ERROR_OUT_OF_MEMORY), Err::STPCUDAError);
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

#define CALC_LAUNCH_CONFIG(VEC) DevLach::determineLaunchConfiguration<Blk>(blockSize, VEC)

template<DevLach::STPDimensionSize Blk, class VecArr>
inline static DevLach::STPLaunchConfiguration computeLaunchConfiguration(const int blockSize, const VecArr& threadDim) {
	switch (threadDim.size()) {
	case 1u:
		return CALC_LAUNCH_CONFIG(threadDim.front());
	case 2u:
		return CALC_LAUNCH_CONFIG(make_vec2(threadDim.data()));
	default:
		return CALC_LAUNCH_CONFIG(make_vec3(threadDim.data()));
	}
}

TEMPLATE_TEST_CASE_SIG("STPDeviceLaunchSetup can automatically configure the best device launch setting for any block dimension",
	"[Utility][STPDeviceLaunchSetup]", ((DevLach::STPDimensionSize BlockDim), BlockDim), 1u, 2u, 3u) {
	
	GIVEN("Desired block and grid dimensions") {
		const auto BlockSize = static_cast<unsigned int>(GENERATE(take(3, range(Info::WarpSize, 32 * Info::WarpSize, Info::WarpSize))));
		const auto ThreadDim = static_cast<DevLach::STPDimensionSize>(GENERATE(values({ 1u, 2u, 3u })));
		const auto ThreadDimVec = GENERATE_COPY(take(3, chunk(ThreadDim, random(1u, 2048u))));

		WHEN("Device launch configuration is calculated based on these values") {
			const auto [LaunchGrid, LaunchBlock] = computeLaunchConfiguration<BlockDim>(BlockSize, ThreadDimVec);

			THEN("The launch configuration calculated is optimal") {
				const auto verifyGridSize = [BlockSize](const auto gridComp, const auto blockComp, const auto threadComp) -> void {
					//the number of thread should be no less than requested
					CHECK(gridComp * blockComp >= threadComp);
					//and there should not be more idling thread than the multiple of block size
					CHECK(gridComp * blockComp - threadComp <= BlockSize);
				};

				//non-zero checking
				REQUIRE((LaunchGrid.x > 0u && LaunchGrid.y > 0u && LaunchGrid.z > 0u));
				REQUIRE((LaunchBlock.x > 0u && LaunchBlock.y > 0u && LaunchBlock.z > 0u));

				//block size checking
				CHECK(LaunchBlock.x * LaunchBlock.y * LaunchBlock.z == BlockSize);
				//grid size checking
				switch (ThreadDimVec.size()) {
				case 3u:
					verifyGridSize(LaunchGrid.z, LaunchBlock.z, ThreadDimVec[2]);
					[[fallthrough]];
				case 2u:
					verifyGridSize(LaunchGrid.y, LaunchBlock.y, ThreadDimVec[1]);
					[[fallthrough]];
				default:
					verifyGridSize(LaunchGrid.x, LaunchBlock.x, ThreadDimVec[0]);
					break;
				}
			}

		}

	}

}