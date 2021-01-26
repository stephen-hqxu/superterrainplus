#include "catch2/catch.hpp"
#include "../Helpers/STPThreadPool.cpp"
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

TEST_CASE("Thread pool functionality test", "[STPThreadPool]") {

	SECTION("Thread pool initialisation") {
		REQUIRE_NOTHROW(STPThreadPool(4u));
	}

	SECTION("Thread pool can run without deadlocking itself") {
		STPThreadPool pool(4u);

		auto start = std::chrono::high_resolution_clock::now();
		REQUIRE_NOTHROW([&pool]() -> void {
			std::future<float> val[16];
			for (int i = 0; i < 16; i++) {
				val[i] = pool.enqueue_future(testing_task, glm::vec4(i));
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
			REQUIRE(pool.isRunning());
			REQUIRE(pool.size() == 0u);
		}

	}

	SECTION("Initialise thread pool with no worker") {
		INFO("It should throw an exception, but it didn't")
		REQUIRE_THROWS_WITH(STPThreadPool(0u), Contains("0"));
	}
}