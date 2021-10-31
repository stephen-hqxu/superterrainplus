#include <SuperTerrain+/Environment/STPConfiguration.h>

using namespace SuperTerrainPlus::STPEnvironment;

using glm::uvec2;
using glm::uvec3;
using glm::vec3;

//STPConfiguration.h

bool STPConfiguration::validate() const {
	return this->ChunkSetting.validate()
		&& this->HeightfieldSetting.validate()
		&& this->MeshSetting.validate();
}

STPChunkSetting& STPConfiguration::getChunkSetting() {
	return this->ChunkSetting;
}

STPHeightfieldSetting& STPConfiguration::getHeightfieldSetting() {
	return this->HeightfieldSetting;
}

STPMeshSetting& STPConfiguration::getMeshSetting() {
	return this->MeshSetting;
}

//STPChunkSetting.h

STPChunkSetting::STPChunkSetting() : STPSetting(), 
	ChunkSize(uvec2(0u)), 
	MapSize(uvec2(0u)), 
	RenderedChunk(uvec2(0u)), 
	ChunkOffset(vec3(0.0f)), 
	ChunkScaling(1.0f), 
	MapOffset(vec3(0.0f)), 
	FreeSlipChunk(uvec2(0u)) {

}

bool STPChunkSetting::validate() const {
	static auto isOdd = [](uvec2 num) constexpr -> bool {
		constexpr uvec2 VecOne = uvec2(1u);
		return (num & VecOne) == VecOne;
	};

	return this->ChunkSize.x > 0u
		&& this->ChunkSize.y > 0u
		&& this->MapSize.x > 0u
		&& this->MapSize.y > 0u
		&& this->RenderedChunk.x > 0u
		&& this->RenderedChunk.y > 0u
		&& this->ChunkScaling > 0.0f
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
	static auto checkRange = [](float value, float lower, float upper) constexpr -> bool {
		return value >= lower && value <= upper;
	};
	//check the raindrop parameter plus also heightmap parameter
	return this->STPRainDropSetting::validate();
}

//STPMeshSetting.h

STPMeshSetting::STPTessellationSetting::STPTessellationSetting() : 
	MaxTessLevel(0.0f), 
	MinTessLevel(0.0f), 
	FurthestTessDistance(0.0f), 
	NearestTessDistance(0.0f) {

}

bool STPMeshSetting::STPTessellationSetting::validate() const {
	return this->MaxTessLevel >= 0.0f
		&& this->MinTessLevel >= 0.0f
		&& this->FurthestTessDistance >= 0.0f
		&& this->NearestTessDistance >= 0.0f
		//range check
		&& this->MaxTessLevel >= this->MinTessLevel
		&& this->FurthestTessDistance >= this->NearestTessDistance;
}

STPMeshSetting::STPMeshSetting() : 
	Strength(1.0f),
	Altitude(1.0f), 
	LoDShiftFactor(2.0f) {

}

bool STPMeshSetting::validate() const {
	return this->Strength > 0.0f 
		&& this->Altitude > 0.0f
		&& this->LoDShiftFactor > 0.0f
		&& this->TessSetting.validate();
}