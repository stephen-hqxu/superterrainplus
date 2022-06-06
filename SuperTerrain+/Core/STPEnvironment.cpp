#include <SuperTerrain+/Environment/STPConfiguration.h>

using namespace SuperTerrainPlus::STPEnvironment;

using glm::uvec2;
using glm::dvec2;
using glm::uvec3;
using glm::dvec3;

//STPConfiguration.h

bool STPConfiguration::validate() const {
	return this->ChunkSetting.validate()
		&& this->HeightfieldSetting.validate();
}

//STPChunkSetting.h

STPChunkSetting::STPChunkSetting() : STPSetting(), 
	ChunkSize(uvec2(0u)), 
	MapSize(uvec2(0u)), 
	RenderedChunk(uvec2(0u)), 
	ChunkOffset(dvec3(0.0)), 
	ChunkScaling(1.0), 
	MapOffset(dvec2(0.0)), 
	FreeSlipChunk(uvec2(0u)) {

}

bool STPChunkSetting::validate() const {
	static constexpr auto isOdd = [](uvec2 num) constexpr -> bool {
		constexpr uvec2 VecOne = uvec2(1u);
		return (num & VecOne) == VecOne;
	};

	return this->ChunkSize.x > 0u
		&& this->ChunkSize.y > 0u
		&& this->MapSize.x > 0u
		&& this->MapSize.y > 0u
		&& this->RenderedChunk.x > 0u
		&& this->RenderedChunk.y > 0u
		&& this->ChunkScaling > 0.0
		&& this->FreeSlipChunk.x > 0u
		&& this->FreeSlipChunk.y > 0u
		//number validation
		&& isOdd(this->RenderedChunk)
		&& isOdd(this->FreeSlipChunk);
}

//STPHeightfieldSetting.h

STPHeightfieldSetting::STPHeightfieldSetting() : STPRainDropSetting(), 
	Seed(0ull) {

}

bool STPHeightfieldSetting::validate() const {
	//check the raindrop parameter plus also heightmap parameter
	return this->STPRainDropSetting::validate();
}

//STPRainDropSetting.h

STPRainDropSetting::STPRainDropSetting() :
	STPSetting(),
	RainDropCount(0u),
	Inertia(0.0f),
	SedimentCapacityFactor(1.0f),
	minSedimentCapacity(0.0f),
	initWaterVolume(1.0f),
	minWaterVolume(0.0f),
	Friction(0.0f),
	initSpeed(0.0f),
	ErodeSpeed(0.0f),
	DepositSpeed(0.0f),
	EvaporateSpeed(0.0f),
	Gravity(1.0f),
	ErosionBrushRadius(0u) {

}

bool STPRainDropSetting::validate() const {
	static constexpr auto checkRange = [](float value, float lower, float upper) constexpr -> bool {
		return value >= lower && value <= upper;
	};

	return checkRange(this->Inertia, 0.0f, 1.0f)
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
		&& this->ErosionBrushRadius != 0u;
}