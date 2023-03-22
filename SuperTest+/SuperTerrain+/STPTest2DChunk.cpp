//Catch2
#include <catch2/catch_test_macros.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

//SuperTerrain+/World/Chunk
#include <SuperTerrain+/World/Chunk/STPChunk.h>

#include <SuperTerrain+/Exception/STPNumericDomainError.h>

//GLM
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>

using namespace SuperTerrainPlus;

using STPDiversity::Sample;

using glm::ivec2;
using glm::uvec2;
using glm::vec2;
using glm::dvec2;
using glm::dvec3;

using std::fill_n;
using std::all_of;

SCENARIO("STPChunk utility functions can compute chunk coordinate correctly", "[Chunk][STPChunk]") {

	GIVEN("A camera position in the world and some chunk parameters") {
		constexpr dvec3 CameraPosition = dvec3(-573.74, 679.5, 845.982);
		constexpr uvec2 ChunkSize = uvec2(8u, 5u);
		constexpr dvec2 ChunkScale = dvec2(2.5, 4.0);

		THEN("The chunk world position should be correctly calculated") {
			constexpr ivec2 ChunkPosition = ivec2(-232, 210);
			CHECK(STPChunk::calcWorldChunkCoordinate(CameraPosition, ChunkSize, ChunkScale) == ChunkPosition);

			WHEN("Trying to generate some map for a chunk with this world coordinate") {

				THEN("Chunk should report the correct map offset to ensure seamless generated texture.") {
					constexpr uvec2 MapSize = uvec2(512u, 256u);
					constexpr dvec2 MapOfffset = dvec2(25.5, -57.5);

					constexpr dvec2 ChunkMapOffset = dvec2(-14822.5, 10694.5);
					CHECK(STPChunk::calcChunkMapOffset(ChunkPosition, ChunkSize, MapSize, MapOfffset) == ChunkMapOffset);
				}

			}
		}

		WHEN("Chunk is offset") {
			constexpr ivec2 ChunkOffset = ivec2(3, -2);

			THEN("The offset chunk world position should be correct") {
				constexpr ivec2 OriginalChunkPosition = ivec2(30, -30);
				constexpr ivec2 OffsetChunkPosition = ivec2(54, -40);
				CHECK(STPChunk::offsetChunk(OriginalChunkPosition, ChunkSize, ChunkOffset) == OffsetChunkPosition);
			}
		}

		WHEN("Asking for a region of chunks") {
			constexpr uvec2 RegionSize = uvec2(5u, 7u);

			THEN("Chunk index can be converted to local coordinate") {
				CHECK(STPChunk::calcLocalChunkCoordinate(7u, RegionSize) == uvec2(2u, 1u));
				CHECK(STPChunk::calcLocalChunkCoordinate(12u, RegionSize) == uvec2(2u));
				CHECK(STPChunk::calcLocalChunkCoordinate(20u, RegionSize) == uvec2(0u, 4u));
			}

			THEN("Local chunk origin can be recovered given centre chunk coordinate") {
				constexpr ivec2 LocalChunkCentre = ivec2(-24, 50);
				CHECK(STPChunk::calcLocalChunkOrigin(LocalChunkCentre, ChunkSize, RegionSize) == ivec2(-40, 35));
			}

			THEN("All chunk neighbour offset within this region should be correct") {
				const auto ChunkRegionOffset = STPChunk::calcChunkNeighbourOffset(ChunkSize, RegionSize);
				//testing every chunk position is too much, let's pick a few
				CHECK(ChunkRegionOffset[0] == ivec2(-16, -15));
				CHECK(ChunkRegionOffset[9] == ivec2(16, -10));
				CHECK(ChunkRegionOffset[17] == ivec2(0));
				CHECK(ChunkRegionOffset[23] == ivec2(8, 5));
				CHECK(ChunkRegionOffset[34] == ivec2(16, 15));
			}
		}
	}
}

class ChunkTester : protected STPChunk {
protected:

	constexpr static uvec2 Size = uvec2(2u);
	constexpr static unsigned int Count = Size.x * Size.y;

	template<typename T>
	inline static void fillValue(T* const texture, const T value) {
		fill_n(texture, ChunkTester::Count, value);
	}

	template<typename T>
	inline static void testMapValue(T* const map, const T reference) {
		CHECK(all_of(map, map + ChunkTester::Count, [reference](auto val) { return val == reference; }));
	}

public:

	ChunkTester() : STPChunk(ChunkTester::Size) {

	}

};

SCENARIO_METHOD(ChunkTester, "STPChunk data structure stores chunk status and texture", "[Chunk][STPChunk]") {

	GIVEN("An invalid chunk object with zero in any of the dimension component") {

		THEN("Construction of such chunk is not allowed") {
			REQUIRE_THROWS_AS(STPChunk(uvec2(0u, 128u)), STPException::STPNumericDomainError);
		}

	}

	GIVEN("A chunk object") {

		THEN("New chunk should be usable and empty with fixed size") {
			REQUIRE(this->Completeness == STPChunk::STPChunkCompleteness::Empty);
			REQUIRE(this->MapDimension == ChunkTester::Size);
		}

		AND_GIVEN("Some texture values") {
			constexpr float float_value = -756.5f;
			constexpr Sample biome_value = 5u;
			constexpr unsigned short buffer_value = 123u;

			WHEN("Trying to write some data into the texture") {
				ChunkTester::fillValue(this->heightmap(), float_value);
				ChunkTester::fillValue(this->biomemap(), biome_value);
				ChunkTester::fillValue(this->heightmapLow(), buffer_value);

				THEN("Value can be retrieved without being corrupted") {
					ChunkTester::testMapValue(this->heightmap(), float_value);
					ChunkTester::testMapValue(this->biomemap(), biome_value);
					ChunkTester::testMapValue(this->heightmapLow(), buffer_value);
				}
			}
		}
		
	}
}