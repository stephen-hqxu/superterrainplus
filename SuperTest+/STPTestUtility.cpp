//Catch2
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

//CUDA
#include "STPTestInformation.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>

//SuperTerrain+/SuperTerrain+/Utility
#include <SuperTerrain+/Utility/STPThreadPool.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>
#include <SuperTerrain+/Utility/Memory/STPMemoryPool.h>
#include <SuperTerrain+/Utility/Memory/STPObjectPool.h>
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceMemory.h>
//SuperTerrain+/SuperTerrain+/Exception
#include <SuperTerrain+/Exception/STPCUDAError.h>
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPDeadThreadPool.h>


#include <algorithm>
#include <optional>

using namespace SuperTerrainPlus;

using std::optional;
using std::unique_ptr;
using std::make_unique;
using std::mutex;
using std::condition_variable;
using std::unique_lock;
using std::future;
using std::async;

class ThreadPoolTester {
protected:

	mutable mutex Lock;
	mutable condition_variable Signal;

	//use optional so we can control the life-time of the thread pool for easy testing
	optional<STPThreadPool> Pool;

	static void sleep(size_t time) {
		std::this_thread::sleep_for(std::chrono::milliseconds(time));
	}

	static unsigned int busyWork(unsigned int value) {
		ThreadPoolTester::sleep(10ull);
		return value;
	}

public:

	ThreadPoolTester() : Pool(1u) {

	}

	void waitUntilSignaled() const {
		{
			unique_lock<mutex> cond_lock(this->Lock);
			this->Signal.wait(cond_lock);
		}
	}

	void sendSignal() const {
		this->Signal.notify_one();
	}

	void killPool() {
		this->Pool.reset();//this line will not return until the previous worker has finished
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
			REQUIRE(this->Pool->isRunning());
			REQUIRE(this->Pool->size() == 0ull);
		}

		WHEN("Some works needs to be done") {
			const unsigned int work = GENERATE(take(3u, random(0u, 6666u)));
			std::future<unsigned int> result;

			THEN("Thread pool should be working on the task") {
				REQUIRE_NOTHROW([&pool = *(this->Pool), &result, work]() {
					result = pool.enqueue_future(&ThreadPoolTester::busyWork, work);
					}());

				AND_THEN("Work can be done successfully") {
					//get the work done
					REQUIRE(result.get() == work);

					//thread pool should be idling
					REQUIRE(this->Pool->size() == 0ull);
				}
			}

		}

		WHEN("Insert more works to a thread pool that has been signalled to be killed") {
			using std::bind;
			//send a busy worker
			this->Pool->enqueue_void(bind(&ThreadPoolTester::waitUntilSignaled, this));
			//send a thread to kill the thread pool
			future<void> killer = async(bind(&ThreadPoolTester::killPool, this));
			
			//destructor of the thread pool will not return until all waiting works are finished
			//wait for the thread to reach the "deadlock" state
			ThreadPoolTester::sleep(25ull);

			THEN("Thread pool should not allow more works to be inserted") {
				CHECK_THROWS_AS(this->Pool->enqueue_void(&ThreadPoolTester::busyWork, 0u), STPException::STPDeadThreadPool);

				//clear up
				this->sendSignal();
				killer.get();
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
				this->release(Mem);
			}

			AND_WHEN("Requesting a memory with size of zero") {
				
				THEN("Error should be thrown to notify that memory size should be positive") {
					REQUIRE_THROWS_AS(this->request(0ull), STPException::STPBadNumericRange);
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
class ObjectPoolTester : protected STPObjectPool<unique_ptr<T>, ObjectCreator<T>> {
public:

	ObjectPoolTester() : STPObjectPool() {

	}

};

TEMPLATE_TEST_CASE_METHOD(ObjectPoolTester, "STPObjectPool reuses object whenever possible", "[Utility][STPObjectPool]",
	float, unsigned int){
	using std::move;

	GIVEN("A fresh object pool") {

		WHEN("An object is requested") {
			auto Obj = this->requestObject();

			THEN("The requested object should be constructed appropriately") {
				REQUIRE(*Obj == static_cast<TestType>(123));
			}

			AND_WHEN("The object is returned and re-requested") {
				this->returnObject(move(Obj));
				auto ReObj = this->requestObject();

				THEN("The requested object should be the same as the first object") {
					REQUIRE(*ReObj == static_cast<TestType>(123));
				}

				AND_WHEN("Another object is requested while the pool is empty") {
					auto AnotherObj = this->requestObject();

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
		unique_ptr<unsigned int[]> HostData = make_unique<unsigned int[]>(8);
		unique_ptr<unsigned int[]> ReturnedData = make_unique<unsigned int[]>(8);

		const unsigned int Data = GENERATE(take(3, random(0u, 66666666u)));
		std::fill_n(HostData.get(), 8, Data);

		WHEN("A smart device memory is requested") {
			constexpr static size_t RowSize2D = sizeof(unsigned int) * 4u;

			auto DeviceData = STPSmartDeviceMemory::makeDevice<unsigned int[]>(8);
			auto StreamedDeviceData = STPSmartDeviceMemory::makeStreamedDevice<unsigned int[]>(STPTestInformation::TestDeviceMemoryPool, 0, 8);
			auto PitchedDeviceData = STPSmartDeviceMemory::makePitchedDevice<unsigned int[]>(4, 2);

			THEN("Smart device memory can be used like normal memory") {
				constexpr static size_t DataSize = sizeof(unsigned int) * 8ull;
				auto compareData = [Data](auto i) {
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