#pragma once

//Catch2
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
//Generators
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

//SuperTerrain+/GPGPU/FreeSlip
#include <SuperTerrain+/GPGPU/FreeSlip/STPFreeSlipGenerator.cuh>

#include <SuperTerrain+/Utility/Exception/STPInvalidArgument.h>
#include <SuperTerrain+/Utility/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Utility/Exception/STPMemoryError.h>

#include <glm/gtc/type_ptr.hpp>

#include <type_traits>
#include <algorithm>
#include <optional>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPCompute;

using STPDiversity::Sample;

using glm::uvec2;

using std::vector;
using std::optional;

template<typename T>
class FreeSlipBufferTester {
protected:

	constexpr static uvec2 SmallDimension = uvec2(4u);
	constexpr static uvec2 SmallChunkUnit = uvec2(2u);
	constexpr static unsigned int SmallSize = SmallDimension.x * SmallDimension.y * SmallChunkUnit.x * SmallChunkUnit.y;

	using CurrentFreeSlipBuffer = STPFreeSlipTextureBuffer<T>;
	using TestData = typename CurrentFreeSlipBuffer::STPFreeSlipTextureData;
	using TestMemoryMode = typename TestData::STPMemoryMode;

	T Reference[SmallSize];
	T Texture[SmallSize];
	vector<T*> TextureBuffer;

	inline static STPFreeSlipTextureAttribute SmallAttribute{ SmallSize, STPPinnedMemoryPool(), 0 };

	static T getRandomData() {
		if constexpr (std::is_floating_point_v<T>) {
			return GENERATE(take(1, random(-66666.0f, 66666.0f)));
		}
		else {
			return GENERATE(take(1, random(0u, 60000u)));
		}
	}

public:

	FreeSlipBufferTester() {
		//fill the reference texture with random value
		std::fill_n(this->Reference, FreeSlipBufferTester<T>::SmallSize, FreeSlipBufferTester<T>::getRandomData());
		
		//texture is the buffer that we are testing
		std::copy_n(this->Reference, FreeSlipBufferTester<T>::SmallSize, this->Texture);

		this->TextureBuffer.emplace_back(this->Texture);
	}

};

#define WRITE_MERGED() \
std::fill_n(Merged, CurrentTester::SmallSize, RandomData); \
REQUIRE_NOTHROW(TestBuffer.reset())
#define MERGED_AVAILABLE() REQUIRE(Merged[index] == this->Texture[index])
#define TEXTURE_WRITTEN() REQUIRE(this->Texture[index] == RandomData)

TEMPLATE_TEST_CASE_METHOD(FreeSlipBufferTester, "STPFreeSlipTextureBuffer can merge and disintegrate per-chunk texture following the memory mode", 
	"[GPGPU][FreeSlip][STPFreeSlipTextureBuffer]", float, Sample) {
	using CurrentTester = FreeSlipBufferTester<TestType>;

	WHEN("Some wrong numbers are provided to the texture buffer") {
		vector<TestType*> EmptyBuffer;
		STPFreeSlipTextureAttribute ZeroPixel = { 0u, { }, 0 };

		THEN("Creation of texture buffer should be prevented") {
			//no texture buffer
			REQUIRE_THROWS_AS(CurrentFreeSlipBuffer(EmptyBuffer, { 1u }, CurrentTester::SmallAttribute), STPException::STPInvalidArgument);
			//no pixel
			REQUIRE_THROWS_AS(CurrentFreeSlipBuffer(this->TextureBuffer, { 1u }, ZeroPixel), STPException::STPBadNumericRange);
			//no channel
			REQUIRE_THROWS_AS(CurrentFreeSlipBuffer(this->TextureBuffer, { }, CurrentTester::SmallAttribute), STPException::STPBadNumericRange);
		}

	}

	GIVEN("A valid array of texture buffer and appropriate memory operation mode") {
		TestMemoryMode Mode = GENERATE(values({
			TestMemoryMode::ReadOnly,
			TestMemoryMode::WriteOnly,
			TestMemoryMode::ReadWrite
		}));
		optional<CurrentFreeSlipBuffer> TestBuffer;

		REQUIRE_NOTHROW([&TestBuffer, Mode, this]() {
			TestData Data = { 1u, Mode, 0 };
			TestBuffer.emplace(this->TextureBuffer, Data, CurrentTester::SmallAttribute);
		}());

		WHEN("Texture buffer is unmerged") {

			THEN("Merge location is not available") {
				REQUIRE_THROWS_AS((*TestBuffer) == STPFreeSlipLocation::HostMemory, STPException::STPMemoryError);
			}

		}

		WHEN("Merge the texture with said memory mode") {
			TestType* Merged;

			THEN("Merging should be successful and return a merged texture") {
				REQUIRE_NOTHROW([&Merged, &TestBuffer]() {
					Merged = (*TestBuffer)(STPFreeSlipLocation::HostMemory);
				}());

				AND_THEN("Merge location is correct and no reallocation when the merging operation is called again, instead the previously returned memory should be returned") {
					REQUIRE((*TestBuffer) == STPFreeSlipLocation::HostMemory);
					REQUIRE((*TestBuffer)(STPFreeSlipLocation::HostMemory) == Merged);
				}

				AND_THEN("Manipulation on the texture data and disintegrating the data repects the memory mode") {
					const auto index = GENERATE(take(3, random(0u, 63u)));
					const TestType RandomData = CurrentTester::getRandomData();

					CHECKED_IF(Mode == TestMemoryMode::ReadOnly) {
						//test if data is available
						MERGED_AVAILABLE();

						//write something to the texture and clean up manually
						WRITE_MERGED();

						//make sure the original data is intact under read only mode
						REQUIRE(this->Reference[index] == this->Texture[index]);
					}
					CHECKED_IF(Mode == TestMemoryMode::WriteOnly) {
						//checking if the merged texture contains garbage data
						//reading garbage data is a undefined behaviour, so good luck
						const bool result = Merged[index] == this->Texture[index];
						CHECK_FALSE(result);
						CHECKED_IF(result) {
							WARN("This assertion involves reading un-initialised memory, re-run this test case a few more times to confirm");
						}

						WRITE_MERGED();

						//the new data should be written back
						TEXTURE_WRITTEN();
					}
					CHECKED_IF(Mode == TestMemoryMode::ReadWrite) {
						MERGED_AVAILABLE();

						WRITE_MERGED();

						TEXTURE_WRITTEN();
					}
				}
			}

		}

	}

}

class LocalIndexRef : protected STPFreeSlipGenerator {
protected:

	constexpr static uvec2 Dimension = uvec2(8u, 4u);
	constexpr static uvec2 ChunkUnit = uvec2(4u, 8u);
	constexpr static uvec2 ChunkRange = Dimension * ChunkUnit;

	inline static STPFreeSlipTextureAttribute IndexAttribute{ Dimension.x * Dimension.y, STPPinnedMemoryPool(), 0 };
	constexpr static STPFreeSlipSampleTextureBuffer::STPFreeSlipTextureData IndexData
		{ 1u, STPFreeSlipSampleTextureBuffer::STPFreeSlipTextureData::STPMemoryMode::ReadOnly, 0 };

public:

	Sample Local[ChunkRange.y][ChunkRange.x];
	vector<Sample*> LocalBuffer;

	LocalIndexRef() : Local(), STPFreeSlipGenerator(LocalIndexRef::ChunkUnit, LocalIndexRef::Dimension) {

		for (unsigned int y = 0u; y < ChunkRange.y; y++) {
			for (unsigned int x = 0u; x < ChunkRange.x; x++) {
				//generate a simple texture with local indices
				this->Local[y][x] = (x + y * ChunkRange.x);
			}
			//push the texture of this chunk
			this->LocalBuffer.emplace_back(Local[y]);
		}
	}

	Sample locate(const uvec2& coordinate) const {
		return this->Local[coordinate.y][coordinate.x];
	}

};

SCENARIO_METHOD(LocalIndexRef, "STPFreeSlipGenerator generates global-local index table and exports data to STPFreeSlipManagerAdaptor", 
	"[GPGPU][FreeSlip][STPFreeSlipGenerator]") {

	WHEN("Invalid data is given to the generator") {
		constexpr uvec2 BadChunkUnit = uvec2(56u, 0u);
		constexpr uvec2 BadDimension = uvec2(8u);

		THEN("Generator should not be created with error thrown") {
			REQUIRE_THROWS_AS(STPFreeSlipGenerator(BadChunkUnit, BadDimension), STPException::STPBadNumericRange);
		}

	}

	GIVEN("A freshly created, valid free-slip generator") {

		THEN("Data stored in the generator should be available upon constrction") {
			REQUIRE(this->getDimension() == LocalIndexRef::Dimension);
			REQUIRE(this->getFreeSlipChunk() == LocalIndexRef::ChunkUnit);
			REQUIRE(this->getFreeSlipRange() == LocalIndexRef::ChunkRange);
		}

		AND_GIVEN("A loaded free-slip texture buffer") {
			STPFreeSlipSampleTextureBuffer TextureBuffer(this->LocalBuffer, LocalIndexRef::IndexData, LocalIndexRef::IndexAttribute);

			WHEN("A free-slip adaptor is requested") {
				auto Adaptor = (*this)(TextureBuffer);

				AND_WHEN("A free-slip manager is generated from the adaptor") {
					auto Manager = Adaptor(STPFreeSlipLocation::HostMemory);
					
					THEN("Data included in the manager should be consistent") {
						REQUIRE(Manager.Data->Dimension == LocalIndexRef::Dimension);
						REQUIRE(Manager.Data->FreeSlipChunk == LocalIndexRef::ChunkUnit);
						REQUIRE(Manager.Data->FreeSlipRange == LocalIndexRef::ChunkRange);

						AND_THEN("Texture can be indexed correctly using global-local index table, and the correctness of index table is verified") {
							const auto IndexXY = GENERATE(take(5, chunk(2, random(0u, 31u))));
							const uvec2 Coordinate = glm::make_vec2(IndexXY.data());
							const unsigned int Local = this->locate(Coordinate);

							//index table correctness, our texture simply converts 2D coordinate to 1D index
							//when texture is flatten in the manager, the relationship is simply:
							//Texture[Index] = Index ==Implies==> Manager[Local] == Texture[Manager(Local)]
							REQUIRE(Manager[Local] == Manager(Local));
							//symmetric, convert to global index and then back to local index
							REQUIRE(Manager[Manager[Local]] == Local);
						}

					}

				}

			}

		}

	}

}