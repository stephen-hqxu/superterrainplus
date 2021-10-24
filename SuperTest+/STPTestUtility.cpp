//Catch2
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

//SuperTerrain+/SuperTerrain+/Utility
#include <SuperTerrain+/Utility/STPThreadPool.h>
#define STP_DEVICE_ERROR_SUPPRESS_CERR
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Utility/STPSmartStream.h>
#include <SuperTerrain+/Utility/STPMemoryPool.h>
#include <SuperTerrain+/Utility/STPSmartDeviceMemory.h>
#include <SuperTerrain+/Utility/STPSmartDeviceMemory.tpp>
//SuperTerrain+/SuperTerrain+/Utility/Exception
#include <SuperTerrain+/Utility/Exception/STPCUDAError.h>
#include <SuperTerrain+/Utility/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Utility/Exception/STPDeadThreadPool.h>

//CUDA
#include "STPTestInformation.h"
#include <cuda.h>
#include <nvrtc.h>

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

		WHEN("Insert more works to a thread pool that has been signaled to be killed") {
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

enum class SmartStreamType : unsigned char {
	Default = 0x00u,
	Priority = 0x01u
};

template<SmartStreamType Pri>
class SmartStreamTester : protected STPSmartStream {
public:

	SmartStreamTester();

};
template<>
SmartStreamTester<SmartStreamType::Default>::SmartStreamTester() : STPSmartStream() {

}

template<>
SmartStreamTester<SmartStreamType::Priority>::SmartStreamTester() : STPSmartStream(cudaStreamDefault, 0u) {

}

TEMPLATE_TEST_CASE_METHOD_SIG(SmartStreamTester, "STPSmartStream manages CUDA stream smartly", "[Utility][STPSmartStream]",
	((SmartStreamType Pri), Pri), SmartStreamType::Default, SmartStreamType::Priority) {

	WHEN("Trying to retrieve CUDA stream priority range") {

		THEN("Value can be retrieved without error") {
			STPSmartStream::STPStreamPriorityRange range;
			REQUIRE_NOTHROW([&range]() -> void {
				range = STPSmartStream::getStreamPriorityRange();
				
			}());
			//verify
			//in fact, CUDA define the "low" value as the greatest priority
			auto [low, high] = range;
			REQUIRE(low <= high);
		}

	}

	GIVEN("A smart stream object") {
		cudaStream_t stream = static_cast<cudaStream_t>(*dynamic_cast<const STPSmartStream*>(this));

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

#define VERIFY_DATA() REQUIRE(std::all_of(ReturnedData.get(), ReturnedData.get() + 8, compareData))

SCENARIO("STPSmartDeviceMemory allocates and auto-delete device pointer", "[Utility][STPSmartDeviceMemory]") {
	
	GIVEN("A piece of host data that is needed to be copied to device memory") {
		unique_ptr<unsigned int[]> HostData = make_unique<unsigned int[]>(8);
		unique_ptr<unsigned int[]> ReturnedData = make_unique<unsigned int[]>(8);

		const unsigned int Data = GENERATE(take(3, random(0u, 66666666u)));
		std::fill_n(HostData.get(), 8, Data);

		WHEN("A smart device memory is requested") {
			auto DeviceData = STPSmartDeviceMemory::makeDevice<unsigned int[]>(8);
			auto StreamedDeviceData = STPSmartDeviceMemory::makeStreamedDevice<unsigned int[]>(STPTestInformation::TestDeviceMemoryPool, 0, 8);

			THEN("Smart device memory can be used like normal memory") {
				constexpr static size_t DataSize = sizeof(unsigned int) * 8ull;
				auto compareData = [Data](auto i) {
					return i == Data;
				};

				//copy back and forth
				//regular device memory
				STPcudaCheckErr(cudaMemcpy(DeviceData.get(), HostData.get(), DataSize, cudaMemcpyHostToDevice));
				STPcudaCheckErr(cudaMemcpy(ReturnedData.get(), DeviceData.get(), DataSize, cudaMemcpyDeviceToHost));
				VERIFY_DATA();

				//stream-ordered device memory
				STPcudaCheckErr(cudaMemcpyAsync(StreamedDeviceData.get(), HostData.get(), DataSize, cudaMemcpyHostToDevice, 0));
				STPcudaCheckErr(cudaMemcpyAsync(ReturnedData.get(), StreamedDeviceData.get(), DataSize, cudaMemcpyDeviceToHost, 0));
				STPcudaCheckErr(cudaStreamSynchronize(0));
				VERIFY_DATA();
			}

		}

	}

}