#include "catch2/catch.hpp"

#include "../World/Chunk/STPChunk.cpp"
#include "../World/Chunk/STPChunkStorage.cpp"

using namespace SuperTerrainPlus;
using namespace Catch::Generators;

GeneratorWrapper<vec2> genVec2() {
	return value(vec2(random(-1000.0f, 1000.0f).get(), random(-1000.0f, 1000.0f).get()));
}

GeneratorWrapper<vec3> genVec3() {
	return value(vec3(random(-1000.0f, 1000.0f).get(), random(-1000.0f, 1000.0f).get(), random(-1000.0f, 1000.0f).get()));
}

TEST_CASE("2D terrain chunk test", "STPChunk") {

	SECTION("Chunk functionality test") {
		
		SECTION("Construction and destruction without errors") {
			REQUIRE_NOTHROW(STPChunk(uvec2(32u)));
		}
	}
}