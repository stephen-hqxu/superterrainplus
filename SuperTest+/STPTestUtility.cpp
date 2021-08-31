#pragma once

//Catch2
#include <catch2/catch_test_macros.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

//SuperTerrain+/Utility
#include <Utility/STPThreadPool.h>

using namespace SuperTerrainPlus;

class ThreadPoolTester {
protected:

	STPThreadPool Pool;

	static unsigned int busyWork(unsigned int value) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		return value;
	}

public:

	ThreadPoolTester() : Pool(2u) {

	}

};

SCENARIO_METHOD(ThreadPoolTester, "STPThreadPool functionality", "[STPThreadPool]") {

	GIVEN("A thread pool object") {

		THEN("New thread pool should be ready to go") {
			REQUIRE(Pool.isRunning());
		}

		THEN("New thread pool should be empty") {
			REQUIRE(Pool.size() == 0ull);
		}

		WHEN("Some works needs to be done") {
			unsigned int work = GENERATE(take(5u, random(0u, 6666u)));
			std::future<unsigned int> result;
			REQUIRE_NOTHROW([&Pool = Pool, &result, work]() {
				result = Pool.enqueue_future(&ThreadPoolTester::busyWork, work);
			}());

			THEN("Later work should have finished with returned result") {
				REQUIRE(result.get() == work);
			}
		}

		THEN("Thread pool should be idling") {
			REQUIRE(Pool.size() == 0ull);
		}
	}
}