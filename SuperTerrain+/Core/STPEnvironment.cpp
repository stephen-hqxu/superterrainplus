#include <SuperTerrain+/Environment/STPChunkSetting.h>
#include <SuperTerrain+/Environment/STPHeightfieldSetting.h>
#include <SuperTerrain+/Environment/STPRainDropSetting.h>

#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

using namespace SuperTerrainPlus::STPEnvironment;

using glm::uvec2;
using glm::dvec2;
using glm::uvec3;
using glm::dvec3;

//STPChunkSetting.h

#define ASSERT_CHUNK(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPChunkSetting)

void STPChunkSetting::validate() const {
	static constexpr auto isOdd = [](const uvec2 num) constexpr noexcept -> bool {
		constexpr uvec2 VecOne = uvec2(1u);
		return (num & VecOne) == VecOne;
	};
	static constexpr auto isPositiveui = [](const uvec2 val) constexpr noexcept -> bool {
		return val.x > 0u && val.y > 0u;
	};
	static constexpr auto isPositived = [](const dvec2 val) constexpr noexcept -> bool {
		return val.x > 0.0 && val.y > 0.0;
	};

	ASSERT_CHUNK(isPositiveui(this->ChunkSize));
	ASSERT_CHUNK(isPositiveui(this->MapSize));
	ASSERT_CHUNK(isPositived(this->ChunkScale));
	//number validation
	ASSERT_CHUNK(isOdd(this->DiversityNearestNeighbour));
	ASSERT_CHUNK(isOdd(this->ErosionNearestNeighbour));
	ASSERT_CHUNK(isOdd(this->RenderDistance));
}

//STPHeightfieldSetting.h

void STPHeightfieldSetting::validate() const {
	//check the raindrop parameter plus also heightmap parameter
	this->Erosion.validate();
}

//STPRainDropSetting.h

#define ASSERT_RAINDROP(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPRainDropSetting)

void STPRainDropSetting::validate() const {
	static constexpr auto checkRange = [](const float value, const float lower, const float upper) constexpr noexcept -> bool {
		return value >= lower && value <= upper;
	};

	ASSERT_RAINDROP(checkRange(this->Inertia, 0.0f, 1.0f));
	ASSERT_RAINDROP(this->SedimentCapacityFactor > 0.0f);
	ASSERT_RAINDROP(this->minSedimentCapacity >= 0.0f);
	ASSERT_RAINDROP(this->initWaterVolume > 0.0f);
	ASSERT_RAINDROP(this->minWaterVolume >= 0.0f);
	ASSERT_RAINDROP(checkRange(this->Friction, 0.0f, 1.0f));
	ASSERT_RAINDROP(this->initSpeed >= 0.0f);
	ASSERT_RAINDROP(checkRange(this->ErodeSpeed, 0.0f, 1.0f));
	ASSERT_RAINDROP(checkRange(this->DepositSpeed, 0.0f, 1.0f));
	ASSERT_RAINDROP(checkRange(this->EvaporateSpeed, 0.0f, 1.0f));
	ASSERT_RAINDROP(this->Gravity > 0.0f);
	ASSERT_RAINDROP(this->ErosionBrushRadius != 0u);
}