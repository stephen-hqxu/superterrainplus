//Catch2
#include <catch2/catch_test_macros.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

//SuperTerrain+/World/Chunk
#include <SuperTerrain+/World/Chunk/STPChunk.h>
#include <SuperTerrain+/World/Chunk/STPChunkStorage.h>

#include <SuperTerrain+/Exception/STPBadNumericRange.h>

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

		THEN("Contruction of such chunk is not allowed") {
			REQUIRE_THROWS_AS(STPChunk(uvec2(0u, 128u)), STPException::STPBadNumericRange);
		}

	}

	GIVEN("A chunk object") {
		constexpr auto no_state = STPChunk::STPChunkState::Empty;

		THEN("New chunk should be usable and empty with fixed size") {
			REQUIRE_FALSE(this->isOccupied());
			REQUIRE(this->getChunkState() == no_state);
			REQUIRE(this->size() == ChunkTester::Size);
		}

		WHEN("Changing the chunk status flags") {

			THEN("Chunk can be locked") {
				//lock the chunk
				this->markOccupancy(true);
				CHECK(this->isOccupied());

				AND_THEN("Chunk can be unlocked") {
					//unlock the chunk
					this->markOccupancy(false);
					CHECK_FALSE(this->isOccupied());
				}
			}

			THEN("Chunk state can be changed randomly") {
				const auto target_state = GENERATE(values({
					STPChunk::STPChunkState::Biomemap_Ready, 
					STPChunk::STPChunkState::Heightmap_Ready, 
					STPChunk::STPChunkState::Erosion_Ready
				}));
				//change chunk status
				this->markChunkState(target_state);
				REQUIRE(this->getChunkState() == target_state);
			}

		}

		AND_GIVEN("Some texture values") {
			constexpr float float_value = -756.5f;
			constexpr Sample biome_value = 5u;
			constexpr unsigned short buffer_value = 123u;

			WHEN("Trying to write some data into the texture") {
				ChunkTester::fillValue(this->getHeightmap(), float_value);
				ChunkTester::fillValue(this->getBiomemap(), biome_value);
				ChunkTester::fillValue(this->getRenderingBuffer(), buffer_value);

				THEN("Value can be retrieved without being corrupted") {
					ChunkTester::testMapValue(this->getHeightmap(), float_value);
					ChunkTester::testMapValue(this->getBiomemap(), biome_value);
					ChunkTester::testMapValue(this->getRenderingBuffer(), buffer_value);
				}
			}
		}
		
	}
}

SCENARIO_METHOD(STPChunkStorage, "STPChunkStorage stores chunks and we can retrieve chunks when needed", "[Chunk][STPChunkStorage]") {

	GIVEN("A chunk storage object and some chunks") {
		constexpr static uvec2 MapSize = uvec2(2u);
		//round to 1 d.p.
		const auto Location = GENERATE(take(3, chunk(2, map<float>([](auto f) { return roundf(f * 10.0f) / 10.0f; }, random(-6666.666f, 6666.666f)))));
		const vec2 InsertLocation = glm::make_vec2(Location.data());

		THEN("A newly created chunk storage is empty") {
			REQUIRE(this->size() == 0ull);
		}

		WHEN("Try to use a non-exisiting chunk on the chunk storage") {

			THEN("Retrival of such chunk returns a null pointer") {
				REQUIRE((*this)[InsertLocation] == nullptr);
			}

			THEN("Removal of such chunk returns failure") {
				REQUIRE_FALSE(this->remove(InsertLocation));
			}

		}

		WHEN("Emplace some new chunks into the chunk storage") {

			THEN("New chunks should have been inserted") {
				const auto [inserted, chunk] = this->construct(InsertLocation, MapSize);
				REQUIRE(inserted);

				AND_WHEN("Insert another chunk with the same world coordinate") {

					THEN("Insertion fails, and the chunk with the coordinate returns") {
						const auto [insertedAgain, chunkAgain] = this->construct(InsertLocation, MapSize);
						REQUIRE_FALSE(insertedAgain);
						//the same pointer as the previous one should be returned
						REQUIRE(chunkAgain == chunk);
					}

				}

				AND_THEN("The inserted chunk can be retrieved") {
					auto* RetrievedChunk = (*this)[InsertLocation];
					REQUIRE(RetrievedChunk == chunk);
				}

				AND_THEN("Chunk storage can be cleared") {
					REQUIRE_NOTHROW(this->clear());
				}

			}

		}

	}

}