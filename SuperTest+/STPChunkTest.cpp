#include "catch2/catch.hpp"

#include "../World/Chunk/STPChunk.cpp"
#include "../World/Chunk/STPChunkStorage.cpp"

#include <fstream>

using namespace SuperTerrainPlus;
using namespace Catch::Generators;

GeneratorWrapper<vec2> genVec2() {
	return value(vec2(random(-1000.0f, 1000.0f).get(), random(-1000.0f, 1000.0f).get()));
}

GeneratorWrapper<vec3> genVec3() {
	return value(vec3(random(-1000.0f, 1000.0f).get(), random(-1000.0f, 1000.0f).get(), random(-1000.0f, 1000.0f).get()));
}

TEST_CASE("2D terrain chunk test", "[STPChunk]") {

	SECTION("Chunk functionality test") {
		
		SECTION("Construction and destruction without errors") {
			REQUIRE_NOTHROW(STPChunk(uvec2(32u)));
		}
	}

	SECTION("Chunk serialisation test") {
		STPChunk chk(uvec2(2u), true);
		float* h = chk.getHeightmap();
		float* n = chk.getNormalmap();
		h[0] = 123.5f;
		h[3] = 456.5f;
		n[0] = -8.5f;
		n[15] = -98.5f;

		SECTION("Serialisation without error") {
			std::fstream saver("../../Saved/test.stp", std::fstream::out | std::fstream::binary | std::fstream::trunc);
			REQUIRE_NOTHROW(saver << &chk);

			saver.close();
		}

		SECTION("Deserialisation to recover data") {
			std::fstream loader("../../Saved/test.stp", std::fstream::in | std::fstream::binary);
			STPChunk* rec = nullptr;
			REQUIRE_NOTHROW(loader >> rec);

			REQUIRE(rec->getSize() == uvec2(2u)); 
			float* h_rec = chk.getHeightmap();
			float* n_rec = chk.getNormalmap();
			REQUIRE(h_rec[0] == h[0] );
			REQUIRE(h_rec[3] == h[3]);
			REQUIRE(n_rec[0] == n[0]);
			REQUIRE(n_rec[15] == n[15]);

			loader.close();
		}

		SECTION("Incompatible major version") {
			std::fstream loader("../../Saved/invalid_version.stp", std::fstream::in | std::fstream::binary);
			STPChunk* err = nullptr;
			REQUIRE_THROWS_AS(loader >> err, STPSerialisationException);
		}

		SECTION("Invalid file format") {
			std::fstream loader("../../Saved/invalid_id.stp", std::fstream::in | std::fstream::binary);
			STPChunk* err = nullptr;
			REQUIRE_THROWS_AS(loader >> err, STPSerialisationException);
		}
	}
}