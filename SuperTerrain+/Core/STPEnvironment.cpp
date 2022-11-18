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

void STPChunkSetting::validate() const {
	static constexpr auto isOdd = [](uvec2 num) constexpr -> bool {
		constexpr uvec2 VecOne = uvec2(1u);
		return (num & VecOne) == VecOne;
	};
	static constexpr auto isPositiveui = [](uvec2 val) constexpr -> bool {
		return val.x > 0u && val.y > 0u;
	};
	static constexpr auto isPositived = [](dvec2 val) constexpr -> bool {
		return val.x > 0.0 && val.y > 0.0;
	};

	if (isPositiveui(this->ChunkSize)
		&& isPositiveui(this->MapSize)
		&& isPositived(this->ChunkScale)
		//number validation
		&& isOdd(this->DiversityNearestNeighbour)
		&& isOdd(this->ErosionNearestNeighbour)
		&& isOdd(this->RenderDistance)) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPChunkSetting validation fails");
}

//STPHeightfieldSetting.h

void STPHeightfieldSetting::validate() const {
	//check the raindrop parameter plus also heightmap parameter
	this->Erosion.validate();
}

//STPRainDropSetting.h

void STPRainDropSetting::validate() const {
	static constexpr auto checkRange = [](float value, float lower, float upper) constexpr -> bool {
		return value >= lower && value <= upper;
	};

	if (checkRange(this->Inertia, 0.0f, 1.0f)
		&& this->SedimentCapacityFactor > 0.0f
		&& this->minSedimentCapacity >= 0.0f
		&& this->initWaterVolume > 0.0f
		&& this->minWaterVolume >= 0.0f
		&& checkRange(this->Friction, 0.0f, 1.0f)
		&& this->initSpeed >= 0.0f
		&& checkRange(this->ErodeSpeed, 0.0f, 1.0f)
		&& checkRange(this->DepositSpeed, 0.0f, 1.0f)
		&& checkRange(this->EvaporateSpeed, 0.0f, 1.0f)
		&& this->Gravity > 0.0f
		&& this->ErosionBrushRadius != 0u) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPRainDropSetting validation fails");
}