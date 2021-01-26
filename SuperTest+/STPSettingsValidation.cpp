#include "catch2/catch.hpp"
#include "../Settings/STPConfigurations.hpp"

using namespace Catch::Generators;
using namespace SuperTerrainPlus::STPSettings;

TEST_CASE("Testing for validation of settings", "[STPConfigurations]") {
	
	SECTION("----- Simplex noise settings validation test -----") {
		STPSimplexNoiseSettings simplex;
		REQUIRE(simplex.validate());

		SECTION("Validation of legal simplex noise settings") {
			auto i = GENERATE(repeat(5, value(0)));
			simplex.Offset = random(0.0, 360.0).get();
			simplex.Distribution = random(1, 32).get();
			REQUIRE(simplex.validate());
		}

		SECTION("Validation of illegal simplex noise settings") {
			auto i = GENERATE(repeat(5, value(0)));
			simplex.Offset = random(360.0, 720.0).get();
			simplex.Distribution = random(-32, 0).get();
			INFO("Simplex noise offset and distribution was: " << simplex.Offset << " and " << simplex.Distribution);
			REQUIRE_FALSE(simplex.validate());

			simplex.Offset = random(-360.0, -1.0).get();
			REQUIRE_FALSE(simplex.validate());
		}
	}

	SECTION("----- Chunk settings validation test -----") {
		STPChunkSettings chunk;
		REQUIRE(chunk.validate());

		SECTION("Validation of legal chunk settings") {
			chunk.ChunkScaling = GENERATE(take(5, random(0.5f, 128.0f)));
			REQUIRE(chunk.validate());
		}

		SECTION("Validation of illegal chunk settings") {
			chunk.ChunkScaling = GENERATE(take(5, random(-128.0f, 0.0f)));
			INFO("Chunk scaling has value of: " << chunk.ChunkScaling);
			REQUIRE_FALSE(chunk.validate());
		}
	}

	SECTION("----- Terrain mesh settings validation test -----") {
		STPMeshSettings mesh;
		REQUIRE(mesh.validate());

		SECTION("Tessellation level bounding check") {
			auto i = GENERATE(repeat(5, value(0)));
			mesh.TessSettings.MinTessLevel = random(1.0f, 32.0f).get();
			mesh.TessSettings.MaxTessLevel = random(1.0f, 32.0f).get();
			mesh.TessSettings.NearestTessDistance = random(20.0f, 500.0f).get();
			mesh.TessSettings.FurthestTessDistance = random(20.0f, 500.0f).get();
			INFO("tess.min: " << mesh.TessSettings.MinTessLevel);
			INFO("tess.max: " << mesh.TessSettings.MaxTessLevel);
			INFO("tess.near: " << mesh.TessSettings.NearestTessDistance);
			INFO("tess.furt: " << mesh.TessSettings.FurthestTessDistance);
			if (mesh.TessSettings.MinTessLevel < mesh.TessSettings.MaxTessLevel && mesh.TessSettings.NearestTessDistance < mesh.TessSettings.FurthestTessDistance) {
				REQUIRE(mesh.validate());
			}
			else {
				REQUIRE_FALSE(mesh.validate());
			}
		}
	}
}