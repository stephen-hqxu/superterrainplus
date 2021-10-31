//Catch2
#include <catch2/catch_test_macros.hpp>
//Generators
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

//SuperTerrain+/Environment
#include <SuperTerrain+/Environment/STPConfiguration.h>
#include <SuperAlgorithm+/STPSimplexNoiseSetting.h>

//Error
#include <SuperTerrain+/Utility/Exception/STPBadNumericRange.h>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPEnvironment;

using glm::vec2;
using glm::uvec2;
using glm::vec3;

SCENARIO_METHOD(STPRainDropSetting, "STPRainDropSetting can generate an erosion brush with indices and weights available on device",
	"[Environment][STPRainDropSetting]") {
	constexpr static uvec2 FreeSlipRange = uvec2(16u);
	constexpr static unsigned int Radius = 4u;
	constexpr static unsigned int PixelCount = FreeSlipRange.x * FreeSlipRange.y;

	GIVEN("A fresh raindrop setting object with erosion brush parameters") {

		WHEN("Parameters given to erosion brush generator is incorrect") {
			
			THEN("Generation should be halted") {
				REQUIRE_THROWS_AS(this->setErosionBrushRadius(FreeSlipRange, 0u), STPException::STPBadNumericRange);
				REQUIRE_THROWS_AS(this->setErosionBrushRadius(uvec2(32u, 0u), Radius), STPException::STPBadNumericRange);
			}

		}

		WHEN("Erosion brush is correctly generated") {
			REQUIRE_NOTHROW(this->setErosionBrushRadius(FreeSlipRange, Radius));

			THEN("Erosion brush information is available on device side") {
				REQUIRE(this->getErosionBrushRadius() == Radius);
				REQUIRE(this->getErosionBrushSize() <= PixelCount);
				//reading the erosion brush indices and weights take too much effort, skip that ;)
			}

		}

	}

}

static void fillHeightfieldSetting(STPHeightfieldSetting* env) {
	env->Seed = 66661313ull;
	//raindrop setting
	env->RainDropCount = 614400u;
	env->Inertia = 0.2f;
	env->SedimentCapacityFactor = 0.5f;
	env->minSedimentCapacity = 0.05f;
	env->initWaterVolume = 1.0f;
	env->minWaterVolume = 0.05f;
	env->Friction = 0.2f;
	env->initSpeed = 0.0f;
	env->ErodeSpeed = 0.25f;
	env->DepositSpeed = 0.25f;
	env->EvaporateSpeed = 0.05f;
	env->Gravity = 9.81f;
}

SCENARIO_METHOD(STPHeightfieldSetting, "STPHeightfieldSetting stores setting for heightfield map generation and simulation", 
	"[Environment][STPHeightfieldSetting]") {

	GIVEN("A heightfield setting object") {
		fillHeightfieldSetting(dynamic_cast<STPHeightfieldSetting*>(this));

		WHEN("Erosion brush is not yet generated") {

			THEN("Setting should not be validated") {
				REQUIRE_FALSE(this->validate());
			}

		}

		REQUIRE_NOTHROW(this->setErosionBrushRadius(uvec2(32u), 8u));

		WHEN("Heightfield setting values are all correct") {

			THEN("Heightfield setting should be validated") {
				REQUIRE(this->validate());
			}

		}

		WHEN("Heightfield setting contains incorrect value(s)") {

			AND_WHEN("Values are not positive") {
				const unsigned char trial = GENERATE(range(0u, 5u));
				switch (trial) {
				case 0u:
					//strength is negative?
					this->minSedimentCapacity = -9.8f;
					break;
				case 1u:
					//water volume is zero?
					this->initWaterVolume = 0.0f;
					break;
				case 2u:
					//gravity is negative?
					this->Gravity = -9.81f;
					break;
				case 3u:
					//evaporate speed is negative?
					this->EvaporateSpeed = -0.5f;
					break;
				case 4u:
					//sediment cap is zero?
					this->SedimentCapacityFactor = 0.0f;
					break;
				default:
					break;
				}

				THEN("Heightfield setting should not be validated") {
					CHECK_FALSE(this->validate());
				}

			}

			AND_WHEN("Values are out-of-bound of defined") {
				const unsigned char trial = GENERATE(range(0u, 3u));
				switch (trial) {
				case 0u:
					//inertia is bigger than 1?
					this->Inertia = 2.5f;
					break;
				case 1u:
					//friction is greater than 1?
					this->Friction = 1.15f;
					break;
				case 2u:
					//erode speed is greater than 1?
					this->ErodeSpeed = 2.5f;
					break;
				default:
					break;
				}

				THEN("Out-of-bound setting should not be validated") {
					CHECK_FALSE(this->validate());
				}

			}

		}

	}

}

static auto& fillMeshSetting(STPMeshSetting* env) {
	//fill in some correct values
	env->Strength = 12.0f;
	env->Altitude = 8848.86f;
	env->LoDShiftFactor = 8.0f;
	auto& tessellation = env->TessSetting;
	tessellation.MinTessLevel = 5.0f;
	tessellation.MaxTessLevel = 32.0f;
	tessellation.NearestTessDistance = 100.0f;
	tessellation.FurthestTessDistance = 1000.0f;

	return tessellation;
}

SCENARIO_METHOD(STPMeshSetting, "STPMeshSetting stores setting for terrain mesh rendering", "[Environment][STPMeshSetting]") {

	GIVEN("A mesh setting object") {
		auto& tessellation = fillMeshSetting(dynamic_cast<STPMeshSetting*>(this));

		WHEN("All values are filled in correctly") {

			THEN("Setting is validated") {
				REQUIRE(this->validate());
			}
		}

		WHEN("Some values don't make sense") {

			AND_WHEN("Values that are required to be positive are negatively filled") {
				const unsigned char trial = GENERATE(range(0u, 3u));
				switch (trial) {
				case 0u:
					//strength is negative?
					this->Strength = -2.5f;
					break;
				case 1u:
					//level is negative?
					tessellation.MinTessLevel = -6.6f;
					break;
				case 2u:
					//zero LoD?
					this->LoDShiftFactor = 0.0f;
					break;
				default:
					break;
				}

				THEN("Values should not be allowed and validation fails") {
					CHECK_FALSE(this->validate());
				}
			}

			AND_WHEN("Ranged values have the boundary swapped") {
				//min is greater than max?
				tessellation.NearestTessDistance = 888.8f;
				tessellation.FurthestTessDistance = 12.5f;

				THEN("Range check should fail") {
					CHECK_FALSE(this->validate());
				}
			}
			
		}

	}

}

static void fillChunkSetting(STPChunkSetting* env) {
	env->ChunkSize = uvec2(6u);
	env->MapSize = uvec2(16u);
	env->RenderedChunk = uvec2(7u);
	env->ChunkOffset = vec3(-5.5f, 0.0f, 9.6f);
	env->ChunkScaling = 6.5f;
	env->MapOffset = vec2(-567.5f, 789.5f);
	env->FreeSlipChunk = uvec2(1u);
}

SCENARIO_METHOD(STPChunkSetting, "STPChunkSetting stores setting for chunk mesh and terrain map generation", "[Environment][STPChunkSetting]") {

	GIVEN("A chunk setting object") {
		fillChunkSetting(dynamic_cast<STPChunkSetting*>(this));

		WHEN("Chunk setting values are filled in correctly") {

			THEN("Chunk setting is validated") {
				REQUIRE(this->validate());
			}

		}

		WHEN("Chunk setting values contain nonsense") {
			const unsigned char trial = GENERATE(range(0u, 4u));
			switch (trial) {
			case 0u:
				//chunk scales to negative?
				this->ChunkScaling = -13.3f;
				break;
			case 1u:
				//map size is zero?
				this->MapSize = uvec2(8u, 0u);
				break;
			case 2u:
				//rendered chunk count is an even number
				this->RenderedChunk = uvec2(6u, 7u);
				break;
			case 3u:
				//free-slip chunk is an even number in both comp?
				this->FreeSlipChunk = uvec2(2u);
				break;
			default:
				break;
			}

			THEN("Setting validation should fail") {
				CHECK_FALSE(this->validate());
			}

		}

	}

}

static void fillSimplexNoiseSetting(STPSimplexNoiseSetting* env) {
	env->Seed = 66661313ull;
	env->Distribution = 12u;
	env->Offset = 3.567f;
}

SCENARIO_METHOD(STPSimplexNoiseSetting, "STPSimplexNoiseSetting stores setting for simplex noise algorithm setup", 
	"[AlgorithmHost][Environment][STPSimplexNoiseSetting]") {

	GIVEN("A simplex noise setting object") {
		fillSimplexNoiseSetting(dynamic_cast<STPSimplexNoiseSetting*>(this));

		WHEN("Simplex noise setting values are correct") {

			THEN("Setting should be validated") {
				REQUIRE(this->validate());
			}

		}

		WHEN("Any give value does not make sense") {
			const unsigned char trial = GENERATE(range(0u, 3u));
			switch (trial) {
			case 0u:
				//distribution is zero?
				this->Distribution = 0u;
				break;
			case 1u:
				//offset is out of bound
				this->Offset = -78.5f;
				break;
			case 2u:
				this->Offset = 415.5f;
				break;
			default:
				break;
			}

			THEN("Setting should not be validated") {
				CHECK_FALSE(this->validate());
			}

		}

	}

}

SCENARIO_METHOD(STPConfiguration, "STPConfiguration stores all environment objects that will be used by the engine", "[Environment][STPConfiguration]") {

	GIVEN("A configuration object") {
		fillChunkSetting(&this->getChunkSetting());
		fillMeshSetting(&this->getMeshSetting());
		fillHeightfieldSetting(&this->getHeightfieldSetting());

		WHEN("Any of the sub-setting is not validated") {
			this->getHeightfieldSetting().minSedimentCapacity = -15.5f;

			THEN("Configuration should not be validated") {
				REQUIRE_FALSE(this->validate());
			}

		}

		WHEN("All setting objects in the configuration are validated") {
			this->getHeightfieldSetting().setErosionBrushRadius(uvec2(64u), 7u);

			THEN("Configuration should be validated") {
				REQUIRE(this->validate());
			}

		}

	}

}