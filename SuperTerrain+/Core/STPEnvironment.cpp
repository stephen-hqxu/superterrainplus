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