//Catch2
#include <catch2/catch_test_macros.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

//SuperTerrain+/World/Chunk
#include <SuperTerrain+/World/Chunk/STPChunk.h>

#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>

//GLM
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>

using namespace SuperTerrainPlus;

using STPDiversity::Sample;

using glm::ivec2;
using glm::uvec2;
using glm::vec2;
using glm::vec3;

SCENARIO("STPChunk static functions can compute chunk coordinate correctly", "[Chunk][STPChunk]") {

	GIVEN("A camera position in the world and some chunk parameters") {
		constexpr vec3 CameraPosition = vec3(-573.74f, 679.5f, 845.982f);
		constexpr uvec2 ChunkSize = uvec2(10u);
		constexpr float ChunkScaling = 25.5f;

		THEN("The chunk world position should be correctly calculated") {
			constexpr vec2 ChunkPosition = vec2(-765.0f, 765.0f);
			CHECK(STPChunk::getChunkPosition(CameraPosition, ChunkSize, ChunkScaling) == ChunkPosition);

			WHEN("Trying to generate some map for a chunk with this world coordinate") {

				THEN("Chunk should report the correct map offset to ensure seamless generated texture.") {
					constexpr uvec2 MapSize = uvec2(512u);
					constexpr vec2 MapOfffset = vec2(25.5f, -57.5f);

					constexpr vec2 ChunkMapOffset = vec2(-1510.5f, 1478.5f);
					CHECK(STPChunk::calcChunkMapOffset(ChunkPosition, ChunkSize, MapSize, MapOfffset, ChunkScaling) == ChunkMapOffset);
				}

			}
		}

		WHEN("Chunk is offset") {
			constexpr ivec2 ChunkOffset = ivec2(3, -2);

			THEN("The offset chunk world position should be correct") {
				constexpr vec2 OriginalChunkPosition = vec2(1000.5f, -672.5f);
				constexpr vec2 OffsetChunkPosition = vec2(1765.5f, -1182.5f);
				CHECK(STPChunk::offsetChunk(OriginalChunkPosition, ChunkSize, ChunkOffset, ChunkScaling) == OffsetChunkPosition);
			}
		}

		WHEN("Asking for a region of chunks") {
			constexpr uvec2 RegionSize = uvec2(5u, 5u);

			THEN("Chunk index can be converted to local coordinate") {
				CHECK(STPChunk::getLocalChunkCoordinate(7u, RegionSize) == uvec2(2u, 1u));
				CHECK(STPChunk::getLocalChunkCoordinate(12u, RegionSize) == uvec2(2u));
				CHECK(STPChunk::getLocalChunkCoordinate(20u, RegionSize) == uvec2(0u, 4u));
			}

			THEN("All chunk world positions within this region should be correct") {
				constexpr vec2 ChunkCentre = vec2(-35.5f, 89.5f);

				const auto ChunkRegionPosition = STPChunk::getRegion(ChunkCentre, ChunkSize, RegionSize, ChunkScaling);
				//testing every chunk position is too much, let's pick a few
				CHECK(ChunkRegionPosition[0] == vec2(-545.5f, -420.5f));
				CHECK(ChunkRegionPosition[9] == vec2(474.5f, -165.5f));
				CHECK(ChunkRegionPosition[12] == ChunkCentre);
				CHECK(ChunkRegionPosition[16] == vec2(-290.5f, 344.5f));
				CHECK(ChunkRegionPosition[24] == vec2(474.5f, 599.5f));
			}
		}
	}
}

class ChunkTester : protected STPChunk {
protected:

	constexpr static uvec2 Size = uvec2(2u);
	constexpr static unsigned int Count = Size.x * Size.y;

	template<typename T>
	static void fillValue(T* texture, T value) {
		for (unsigned int y = 0u; y < ChunkTester::Size.y; y++) {
			for (unsigned int x = 0u; x < ChunkTester::Size.x; x++) {
				texture[x + y * ChunkTester::Size.x] = value;
			}
		}
	}

	template<typename T>
	inline static void testMapValue(T* map, T reference) {
		CHECK(std::all_of(map, map + ChunkTester::Count, [reference](auto val) { return val == reference; }));
	}

public:

	ChunkTester() : STPChunk(ChunkTester::Size) {

	}

};

SCENARIO_METHOD(ChunkTester, "STPChunk stores chunk status and texture", "[Chunk][STPChunk]") {

	GIVEN("An invalid chunk object with zero in any of the dimension component") {

		THEN("Construction of such chunk is not allowed") {
			REQUIRE_THROWS_AS(STPChunk(uvec2(0u, 128u)), STPException::STPBadNumericRange);
		}

	}

	GIVEN("A chunk object") {
		constexpr auto no_state = STPChunk::STPChunkState::Empty;

		THEN("New chunk should be usable and empty with fixed size") {
			REQUIRE_FALSE(this->occupied());
			REQUIRE(this->chunkState() == no_state);
			REQUIRE(this->PixelSize == ChunkTester::Size);
		}

		WHEN("Try to visit the chunk") {
			STPChunk::STPSharedMapVisitor SharedVisitor(*this);

			AND_WHEN("No unique visitor is alive") {

				THEN("Visit from shared visitor is allowed") {
					REQUIRE_NOTHROW(SharedVisitor.heightmap());
					REQUIRE_FALSE(this->occupied());
				}

				THEN("Unique visitor can be created without problem") {
					STPChunk::STPUniqueMapVisitor UniqueVisitor(*this);

					REQUIRE_NOTHROW(UniqueVisitor.biomemap());
					REQUIRE(this->occupied());
				}

			}
			
			AND_WHEN("Unique visitor is alive") {
				STPChunk::STPUniqueMapVisitor UniqueVisitor(*this);
				REQUIRE(this->occupied());

				THEN("Visit from shared visitor is prohibited") {
					REQUIRE_THROWS_AS(SharedVisitor.heightmapBuffer(), STPException::STPMemoryError);
				}

				THEN("Multiple alive unique visitor is not allowed") {
					REQUIRE_THROWS_AS([this]() {
						STPChunk::STPUniqueMapVisitor UniqueVisitor(*this);
					}(), STPException::STPMemoryError);
				}

			}

		}

		WHEN("Changing the chunk status flags") {

			THEN("Chunk state can be changed randomly") {
				const auto target_state = GENERATE(values({
					STPChunk::STPChunkState::BiomemapReady, 
					STPChunk::STPChunkState::HeightmapReady, 
					STPChunk::STPChunkState::Complete
				}));
				//change chunk status
				this->markChunkState(target_state);
				REQUIRE(this->chunkState() == target_state);
			}

		}

		AND_GIVEN("Some texture values") {
			constexpr float float_value = -756.5f;
			constexpr Sample biome_value = 5u;
			constexpr unsigned short buffer_value = 123u;

			WHEN("Trying to write some data into the texture") {
				STPChunk::STPUniqueMapVisitor Visitor(*this);

				ChunkTester::fillValue(Visitor.heightmap(), float_value);
				ChunkTester::fillValue(Visitor.biomemap(), biome_value);
				ChunkTester::fillValue(Visitor.heightmapBuffer(), buffer_value);

				THEN("Value can be retrieved without being corrupted") {
					ChunkTester::testMapValue(Visitor.heightmap(), float_value);
					ChunkTester::testMapValue(Visitor.biomemap(), biome_value);
					ChunkTester::testMapValue(Visitor.heightmapBuffer(), buffer_value);
				}
			}
		}
		
	}
}