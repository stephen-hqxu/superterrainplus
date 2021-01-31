#include "catch2/catch.hpp"
#include "../Helpers/STPThreadPool.cpp"
#include "../Helpers/STPMemoryPool.hpp"
#include "glm/gtc/noise.hpp"

#include <chrono>

using namespace Catch::Generators;
using Catch::Matchers::EndsWith;
using Catch::Matchers::Contains;
using namespace SuperTerrainPlus;

float testing_task(glm::vec4 input) {
	float res = 0.0f;
	for (int i = 0; i < 10240; i++) {
		res += glm::simplex(input);
	}
	return res;
}

class threadpool_test : public STPThreadPool {
public:

	threadpool_test() : STPThreadPool(4u) {

	}

};

class test_allocator {
public:

	float* allocate(size_t count) {
		//size must be the same
		return reinterpret_cast<float*>(malloc(count * sizeof(float)));
	}

	void deallocate(size_t count, float* ptr) {
		free(ptr);
	}
};

class memorypool_test : public STPMemoryPool<float, test_allocator> {
public:



};

TEST_CASE_METHOD(threadpool_test, "Thread pool functionality test", "[STPThreadPool]") {

	SECTION("Thread pool can run without deadlocking itself") {
		auto start = std::chrono::high_resolution_clock::now();
		REQUIRE_NOTHROW([&]() -> void {
			std::future<float> val[16];
			for (int i = 0; i < 16; i++) {
				val[i] = enqueue_future(testing_task, glm::vec4(i));
			}
			for (int i = 0; i < 16; i++) {
				val[i].get();
			}
		}());
		auto stop = std::chrono::high_resolution_clock::now();
		const long long runtime = std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count();

		if (runtime > 1000ll) {
			WARN("Slow performance on threadpool, runtime was: " << runtime << "ms");
		}

		SECTION("Thread pool flag can be identified") {
			REQUIRE(isRunning());
			REQUIRE(size() == 0u);
		}

	}

	SECTION("Initialise thread pool with no worker") {
		INFO("It should throw an exception, but it didn't")
		REQUIRE_THROWS_WITH(STPThreadPool(0u), Contains("0"));
	}
}

TEST_CASE_METHOD(memorypool_test, "Memory pool functionality test", "[STPMemoryPool]") {
	size_t count = 5ull;

	SECTION("Basic memory allocation and deallocation") {
		float* ptr = allocate(count);
		deallocate(count, ptr);
		REQUIRE(size() == 1);
		REQUIRE(size(count) == 1);
		this->free(count);
	}

	SECTION("Memory size check") {
		float* ptr = allocate(count);
		REQUIRE(empty());
		REQUIRE(empty(count));

		deallocate(count, ptr);
		REQUIRE(size() == 1);
		REQUIRE(size(count) == 1);

		ptr = allocate(count);
		REQUIRE(empty());
		REQUIRE(empty(count));

		this->free(count, ptr);
		REQUIRE(empty());
		REQUIRE(empty(count));
	}

	SECTION("Memory free up") {
		REQUIRE(empty());
		REQUIRE(empty(count));

		float* ptr = allocate(count);
		this->free(count, ptr);
		REQUIRE(empty());
		REQUIRE(empty(count));

		ptr = allocate(count);
		deallocate(count, ptr);
		ptr = allocate(count);
		this->free(count, ptr);
		REQUIRE(empty());
		REQUIRE(empty(count));

		shrink_to_fit();
	}

	SECTION("Memory reallocation and reuse") {
		float* ptr = allocate(count);
		deallocate(count, ptr);
		REQUIRE(ptr == allocate(count));
		deallocate(count, ptr);

		REQUIRE(size() == 1);
		REQUIRE(size(count) == 1);
	}

	SECTION("Allocating different sizes of memory") {
		float* ptr1 = allocate(5u);
		deallocate(5u, ptr1);
		REQUIRE(ptr1 == allocate(5u));
		deallocate(5u, ptr1);

		float* ptr2 = allocate(10u);
		deallocate(10u, ptr2);
		REQUIRE(ptr2 == allocate(10u));
		deallocate(10u, ptr2);

		REQUIRE(size() == 2);
		REQUIRE(size(5u) == 1);
		REQUIRE(size(10u) == 1);
		this->free();
	}
}