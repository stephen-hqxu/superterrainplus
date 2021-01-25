#include "catch2/catch.hpp"
#include "../Settings/STPConfigurations.hpp"

using namespace Catch::Generators;
using namespace SuperTerrainPlus::STPSettings;

TEST_CASE("Testing for validation of settings", "[STPConfigurations]") {
	
	SECTION("Simplex noise settings validation test") {
		STPSimplexNoiseSettings simplex;

		INFO("Validation of default settings");
		REQUIRE(simplex.validate());

		SECTION("Validation of legal settings") {
			auto i = GENERATE(repeat(5, value(0)));
			simplex.Offset = random(0.0, 360.0).get();
			simplex.Distribution = random(1, 32).get();
			REQUIRE(simplex.validate());
		}

		SECTION("Validation of illegal settings") {
			auto i = GENERATE(repeat(5, value(0)));
			simplex.Offset = random(360.0, 720.0).get();
			simplex.Distribution = random(-32, 0).get();
			REQUIRE_FALSE(simplex.validate());

			INFO("Having negative offset values");
			simplex.Offset = random(-360.0, -1.0).get();
			REQUIRE_FALSE(simplex.validate());
		}
	}

	SECTION("Chunk settings validation test") {
		STPChunkSettings chunk;

		INFO("Validation of default settings");
		REQUIRE(chunk.validate());

		SECTION("Validation of legal settings") {
			chunk.ChunkScaling = GENERATE(take(5, random(0.5f, 128.0f)));
			REQUIRE(chunk.validate());
		}

		SECTION("Validation of illegal settings") {
			INFO("Having negative scaling factor");
			chunk.ChunkScaling = GENERATE(take(5, random(-128.0f, 0.0f)));
			REQUIRE_FALSE(chunk.validate());
		}
	}
}